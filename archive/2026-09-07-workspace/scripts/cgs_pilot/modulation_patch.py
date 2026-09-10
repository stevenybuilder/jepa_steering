#!/usr/bin/env python3
"""E2: action-entry path patching on the AdaLN modulation vectors (design doc
"JEPA rep geometry.md", sections 1, 2.2 and 4/E2).

The action exists inside the JEPA-WM predictor only as the 12 x 3 = 36 AdaLN
modulation vectors ``(shift, scale, gate)`` for attention and MLP in each block
(``predictor_blocks[b].adaLN_modulation(action_encoder(a))`` -> ``[B, T, 6D]`` in
the order shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
broadcast to all tokens).  For every block ``b`` and component set
``{attn, mlp, all}`` we run the predictor on the ``a0`` cells (H0, H1, H0') with
the modulation of block ``b`` taken from ``a1`` and every other block from ``a0``
(imagined step 0), and measure

- route-token (gripper + corridor) and egg-token change ``dP_b`` vs the full
  ``a0 -> a1`` change (fraction of norm, cosine);
- the block-induced interaction ``I_b = dP_b(H1) - dP_b(H0)`` on route tokens:
  recovery of ``I_true`` (projection ``<I_b, I_true> / ||I_true||^2``; target
  >= 30 %), cosine with ``I_true`` (> 0.5), and the H0' donor recovery
  ``<dP_b(H0') - dP_b(H0), I_true> / ||I_true||^2`` (< 10 %);
- controls: gate-only sham (block ``b`` re-injected with its own ``a0``
  modulation -> zero change), the full 12-block swap (ceiling; identical to
  running with ``a1``), the single-block swap on H0';
- the direct sensitivity ``dP_g / dmod_b`` (3 x 1024 dims per component set) by
  forward-AD tangent injection at the modulation output: the directional
  derivative along ``mod_b(a1) - mod_b(a0)`` (its route norm per cell and its
  hazard dependence; also the linearisation error of the finite swap) and a
  Hutchinson estimate of the Frobenius norm over ``--n-probes`` random unit
  tangents.

Statistics: unit = scene; scene-clustered bootstrap CIs; sign-flip over scenes of
``recovery - H0' recovery`` with max-T over blocks (per component set).
Decision rule (E2): a single block with recovery >= 0.30, H0' < 0.10 and cosine
> 0.5 localises the entry of the relational computation; every single block
< 0.10 while the full swap >= 0.30 is a distributed-entry result.

Output ``<out>/modulation_patch.json`` with a per-block table.

References: causal mediation on the conditioning pathway (Vig et al.,
arXiv:2004.12265); activation-patching best practices (arXiv:2309.16042);
JEPA-WM (AdaLN action conditioning).
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
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_jacobian_sonar import _predict, _sdpa_math  # noqa: E402
from cgs_stats import cluster_bootstrap, json_safe, sign_flip_maxt  # noqa: E402
from localize_interaction import NULL_CELLS, encode_cell, merge_manifests, null_factor_pairs, null_factor_status, select_pairs  # noqa: E402
from model_action_sensitivity import load_model  # noqa: E402
from predictor_hooks import ZERO_SHOT_SCOPE, n_layers_of, predictor_of, spatial_tokens_of  # noqa: E402
from protocol import CELL_ORDER, canonical_json, sha256_file  # noqa: E402
from token_groups import CellGroups, load_cell_groups, union  # noqa: E402


COMPONENTS = ("attn", "mlp", "all")


def component_slice(comp: str, D: int) -> slice:
    return {"attn": slice(0, 3 * D), "mlp": slice(3 * D, 6 * D), "all": slice(0, 6 * D)}[comp]


class ModulationHooks:
    """Capture and/or replace AdaLN modulation outputs ``[B, T, 6D]`` at one imagined step.

    ``swaps[b] = (comp, values)`` writes ``values[..., slice]`` into block b's output;
    ``tangents[b] = (comp, tangent)`` attaches a forward-mode tangent (zero elsewhere).
    """

    def __init__(self, predictor: torch.nn.Module, swaps: dict[int, tuple[str, torch.Tensor]] | None = None,
                 tangents: dict[int, tuple[str, torch.Tensor]] | None = None, capture: bool = False, target_step: int = 0) -> None:
        self.predictor = predictor
        self.swaps = swaps or {}
        self.tangents = tangents or {}
        self.capture = capture
        self.captured: dict[int, torch.Tensor] = {}
        self.target_step = target_step
        self.step = -1
        self._handles: list[Any] = []

    def _hook(self, b: int, out: torch.Tensor):
        if self.step != self.target_step:
            return out
        D6 = out.shape[-1]
        D = D6 // 6
        if self.capture:
            primal = fwAD.unpack_dual(out).primal if fwAD.unpack_dual(out).tangent is not None else out
            self.captured[b] = primal.detach().clone()
        result = out
        if b in self.swaps:
            comp, values = self.swaps[b]
            result = out.clone()
            result[..., component_slice(comp, D)] = values[..., component_slice(comp, D)].to(out.dtype)
        if b in self.tangents:
            comp, tangent = self.tangents[b]
            primal = fwAD.unpack_dual(result).primal
            full = torch.zeros_like(primal)
            full[..., component_slice(comp, D)] = tangent.to(primal.dtype)
            result = fwAD.make_dual(primal, full)
        return result

    def __enter__(self) -> "ModulationHooks":
        def count(module, args, kwargs):
            self.step += 1

        self._handles.append(self.predictor.register_forward_pre_hook(count, with_kwargs=True))
        for b, blk in enumerate(self.predictor.predictor_blocks):
            self._handles.append(blk.adaLN_modulation.register_forward_hook(lambda m, a, out, b=b: self._hook(b, out)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def capture_modulations(wm: Any, predictor: torch.nn.Module, z: torch.Tensor, a: torch.Tensor, k: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    hooks = ModulationHooks(predictor, capture=True, target_step=k)
    with torch.no_grad(), hooks:
        pred = _predict(wm, z, a, k)
    return pred, hooks.captured


def predict_with_swaps(wm: Any, predictor: torch.nn.Module, z: torch.Tensor, a: torch.Tensor, k: int, swaps: dict[int, tuple[str, torch.Tensor]]) -> torch.Tensor:
    with torch.no_grad(), ModulationHooks(predictor, swaps=swaps, target_step=k):
        return _predict(wm, z, a, k)


def modulation_jvp(wm: Any, predictor: torch.nn.Module, z: torch.Tensor, a: torch.Tensor, k: int, b: int, comp: str, tangent: torch.Tensor) -> torch.Tensor:
    """``(dP / dmod_b[comp]) . tangent`` with ``tangent`` shaped like the component slice ``[B, T, 3D or 6D]``."""

    with _sdpa_math(), fwAD.dual_level(), ModulationHooks(predictor, tangents={b: (comp, tangent)}, target_step=k):
        pred = _predict(wm, z, a, k)
        tg = fwAD.unpack_dual(pred).tangent
    return torch.zeros_like(pred) if tg is None else tg.detach().clone()


# --------------------------------------------------------------------------- #
# Per-scene evaluation
# --------------------------------------------------------------------------- #


def _proj(x: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    xf, rf = x.flatten().double(), ref.flatten().double()
    return {
        "recovery": float((xf @ rf).item() / max(float((rf @ rf).item()), 1e-12)),
        "cosine": float((xf @ rf / (xf.norm() * rf.norm()).clamp_min(1e-12)).item()),
        "norm": float(xf.norm().item()),
    }


def evaluate_scene(
    wm: Any,
    predictor: torch.nn.Module,
    latents: dict[tuple[int, int], dict[str, torch.Tensor]],
    groups: dict[tuple[int, int], CellGroups],
    k: int,
    comps: tuple[str, ...],
    n_probes: int,
    generator: torch.Generator,
    blocks: list[int] | None = None,
) -> dict[str, Any]:
    keys = list(latents)
    has_null = all(kk in latents for kk in NULL_CELLS)
    hazards = (0, 1, 2) if has_null else (0, 1)
    D = latents[keys[0]]["z_context"].shape[-1]
    n_blocks = len(predictor.predictor_blocks)
    use_blocks = list(range(n_blocks)) if blocks is None else [b for b in blocks if b < n_blocks]

    def z_of(h):
        return latents[(h, 0)]["z_context"].detach().clone()

    a0 = latents[(0, 0)]["actions"].detach().clone()
    a1 = latents[(0, 1)]["actions"].detach().clone()
    out_frame = k + 1
    G = union(*(groups[kk].group(out_frame, g) for kk in keys for g in ("gripper", "corridor")))
    E = union(*(groups[kk].group(out_frame, "egg") for kk in keys))
    fut = {kk: latents[kk]["z_future"][0, k].reshape(-1, D).float() for kk in CELL_ORDER}
    I_true = fut[(1, 1)] - fut[(1, 0)] - fut[(0, 1)] + fut[(0, 0)]
    Gi = torch.as_tensor(G if len(G) else np.arange(I_true.shape[0]), dtype=torch.long, device=I_true.device)
    Ei = torch.as_tensor(E, dtype=torch.long, device=I_true.device) if len(E) else Gi

    # clean runs + modulation capture for a0 and a1 on every hazard context
    P0, mod0, P1, mod1 = {}, {}, {}, {}
    for h in hazards:
        P0[h], mod0[h] = capture_modulations(wm, predictor, z_of(h), a0, k)
        P1[h], mod1[h] = capture_modulations(wm, predictor, z_of(h), a1, k)
    full = {h: P1[h] - P0[h] for h in hazards}
    # sanity: swapping all blocks reproduces the a1 run (actions enter only through the modulations)
    full_swap = predict_with_swaps(wm, predictor, z_of(0), a0, k, {b: ("all", mod1[0][b]) for b in range(n_blocks)}) - P0[0]
    full_swap_error = float(((full_swap - full[0]).norm() / full[0].norm().clamp_min(1e-12)).item())
    I_full = full[1] - full[0]
    ceiling = {"route": _proj(I_full[Gi], I_true[Gi]), "egg": _proj(I_full[Ei], I_true[Ei]),
               "null_route": _proj((full[2] - full[0])[Gi], I_true[Gi]) if has_null else None}

    blocks_out: dict[str, Any] = {}
    for b in use_blocks:
        for comp in comps:
            dP = {h: predict_with_swaps(wm, predictor, z_of(h), a0, k, {b: (comp, mod1[h][b])}) - P0[h] for h in hazards}
            sham = predict_with_swaps(wm, predictor, z_of(0), a0, k, {b: (comp, mod0[0][b])}) - P0[0]
            I_b = dP[1] - dP[0]
            entry: dict[str, Any] = {
                "block": b, "component": comp,
                "sham_norm": float(sham.norm().item()),
                "route_fraction_of_full": {f"h{h}": float((dP[h][Gi].norm() / full[h][Gi].norm().clamp_min(1e-12)).item()) for h in hazards},
                "egg_fraction_of_full": {f"h{h}": float((dP[h][Ei].norm() / full[h][Ei].norm().clamp_min(1e-12)).item()) for h in hazards},
                "route_cosine_with_full": {f"h{h}": float(torch.nn.functional.cosine_similarity(dP[h][Gi].flatten(), full[h][Gi].flatten(), dim=0).item()) for h in hazards},
                "route_delta_norm": {f"h{h}": float(dP[h][Gi].norm().item()) for h in hazards},
                "interaction_route": _proj(I_b[Gi], I_true[Gi]),
                "interaction_egg": _proj(I_b[Ei], I_true[Ei]),
                "interaction_route_fraction_of_full_interaction": float((I_b[Gi].norm() / I_full[Gi].norm().clamp_min(1e-12)).item()),
            }
            if has_null:
                entry["null_interaction_route"] = _proj((dP[2] - dP[0])[Gi], I_true[Gi])
            entry["recovery_minus_null"] = entry["interaction_route"]["recovery"] - (entry["null_interaction_route"]["recovery"] if has_null else 0.0)
            # direct sensitivity dP_g/dmod_b: directional derivative along mod1 - mod0 and Hutchinson Frobenius estimate
            sl = component_slice(comp, D)
            sens = {}
            for h in hazards:
                dmod = (mod1[h][b] - mod0[h][b])[..., sl]
                jv = modulation_jvp(wm, predictor, z_of(h), a0, k, b, comp, dmod)
                probes = []
                for _ in range(n_probes):
                    r = torch.randn(dmod.shape, generator=generator, device=dmod.device, dtype=dmod.dtype)
                    r = r / r.norm().clamp_min(1e-12)
                    probes.append(float(modulation_jvp(wm, predictor, z_of(h), a0, k, b, comp, r)[Gi].norm().item() ** 2))
                dim = int(dmod.numel())
                sens[f"h{h}"] = {
                    "directional_jvp_route_norm": float(jv[Gi].norm().item()),
                    "delta_mod_norm": float(dmod.norm().item()),
                    "linearisation_error_route": float(((dP[h] - jv)[Gi].norm() / dP[h][Gi].norm().clamp_min(1e-12)).item()),
                    "frobenius_estimate_route": float(np.sqrt(np.mean(probes) * dim)) if probes else None,
                    "n_probes": n_probes, "dim": dim,
                }
            entry["direct_sensitivity"] = sens
            entry["direct_sensitivity_hazard_dependence_route"] = sens["h1"]["directional_jvp_route_norm"] - sens["h0"]["directional_jvp_route_norm"]
            if has_null:
                entry["direct_sensitivity_null_dependence_route"] = sens["h2"]["directional_jvp_route_norm"] - sens["h0"]["directional_jvp_route_norm"]
            blocks_out[f"b{b:02d}|{comp}"] = entry
    return {"step": k, "route_size": int(len(G)), "egg_size": int(len(E)), "null_factor": has_null, "I_true_route_norm": float(I_true[Gi].norm().item()),
            "full_swap_error": full_swap_error, "ceiling": ceiling, "blocks": blocks_out}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(per_scene: dict[str, dict[str, Any]], comps: tuple[str, ...], n_boot: int, n_perm: int, rng: np.random.Generator, alpha: float,
              recovery_threshold: float = 0.30, null_threshold: float = 0.10, cosine_threshold: float = 0.5) -> dict[str, Any]:
    pair_ids = sorted(per_scene)
    keys = sorted({k for p in pair_ids for k in per_scene[p]["blocks"]})
    table: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    for comp in comps:
        vectors = {}
        for key in keys:
            if not key.endswith(f"|{comp}"):
                continue
            rows = [per_scene[p]["blocks"][key] for p in pair_ids if key in per_scene[p]["blocks"]]
            entry = {"block": rows[0]["block"], "component": comp, "n_scenes": len(rows)}
            for field, getter in (
                ("recovery", lambda r: r["interaction_route"]["recovery"]),
                ("cosine", lambda r: r["interaction_route"]["cosine"]),
                ("null_recovery", lambda r: r["null_interaction_route"]["recovery"] if "null_interaction_route" in r else np.nan),
                ("recovery_minus_null", lambda r: r["recovery_minus_null"]),
                ("egg_recovery", lambda r: r["interaction_egg"]["recovery"]),
                ("route_fraction_of_full_h1", lambda r: r["route_fraction_of_full"]["h1"]),
                ("egg_fraction_of_full_h1", lambda r: r["egg_fraction_of_full"]["h1"]),
                ("interaction_fraction_of_full_interaction", lambda r: r["interaction_route_fraction_of_full_interaction"]),
                ("sham_norm", lambda r: r["sham_norm"]),
                ("directional_jvp_route_norm_h1", lambda r: r["direct_sensitivity"]["h1"]["directional_jvp_route_norm"]),
                ("direct_sensitivity_hazard_dependence_route", lambda r: r["direct_sensitivity_hazard_dependence_route"]),
                ("linearisation_error_route_h1", lambda r: r["direct_sensitivity"]["h1"]["linearisation_error_route"]),
                ("frobenius_estimate_route_h1", lambda r: r["direct_sensitivity"]["h1"]["frobenius_estimate_route"] or np.nan),
            ):
                vals = np.asarray([getter(r) for r in rows], dtype=float)
                entry[field] = cluster_bootstrap(vals, n_boot, rng)
                entry[field]["per_scene"] = {p: (None if np.isnan(v) else float(v)) for p, v in zip([p for p in pair_ids if key in per_scene[p]["blocks"]], vals)}
            v = np.asarray([r["recovery_minus_null"] for r in rows]).reshape(-1, 1)
            if len(v) == len(pair_ids):
                vectors[key] = v
            table[key] = entry
        if vectors:
            res = sign_flip_maxt(vectors, n_perm, rng, statistic=lambda v, s: v[:, 0] if s is None else v[:, 0] * s)
            tests[comp] = {k: v for k, v in res.items() if k != "sites"}
            for key, test in res["sites"].items():
                table[key]["recovery_minus_null_test"] = test
    ceiling = {}
    for field in ("route", "egg", "null_route"):
        vals = np.asarray([per_scene[p]["ceiling"][field]["recovery"] if per_scene[p]["ceiling"][field] else np.nan for p in pair_ids], dtype=float)
        ceiling[f"{field}_recovery"] = cluster_bootstrap(vals, n_boot, rng)
        vals = np.asarray([per_scene[p]["ceiling"][field]["cosine"] if per_scene[p]["ceiling"][field] else np.nan for p in pair_ids], dtype=float)
        ceiling[f"{field}_cosine"] = cluster_bootstrap(vals, n_boot, rng)
    for key, entry in table.items():
        med = lambda f: float(np.nanmedian([v for v in entry[f]["per_scene"].values() if v is not None])) if any(v is not None for v in entry[f]["per_scene"].values()) else float("nan")  # noqa: E731
        p = entry.get("recovery_minus_null_test", {}).get("p_maxt_fwer")
        entry["E2_single_block"] = bool(med("recovery") >= recovery_threshold and (np.isnan(med("null_recovery")) or med("null_recovery") < null_threshold)
                                        and med("cosine") > cosine_threshold and p is not None and p == p and p < alpha)
    full_ok = bool(ceiling["route_recovery"]["point"] >= recovery_threshold)
    singles_below = {comp: all(float(np.nanmedian([v for v in e["recovery"]["per_scene"].values() if v is not None])) < null_threshold for e in table.values() if e["component"] == comp) for comp in comps}
    verdict = {"E2_sites": [k for k, e in table.items() if e["E2_single_block"]], "full_swap_recovery_ge_threshold": full_ok,
               "distributed_entry": {comp: bool(full_ok and singles_below[comp]) for comp in comps}}
    return {"table": table, "permutation": tests, "ceiling": ceiling, "verdict": verdict}


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
    parser.add_argument("--model-label", default="JEPA-WM DROID modulation (action-entry) patching")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--components", nargs="*", default=list(COMPONENTS))
    parser.add_argument("--blocks", type=int, nargs="*", default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--n-probes", type=int, default=8)
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--allow-calibration-seeds", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor = predictor_of(wm)
    for p in wm.parameters():
        p.requires_grad_(False)
    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds)
    if not pairs:
        raise SystemExit("no complete quartets selected")
    null_status = null_factor_status(pairs)
    null_pairs = set(null_factor_pairs(pairs))
    comps = tuple(c for c in args.components if c in COMPONENTS)

    per_scene: dict[str, dict[str, Any]] = {}
    for pair_id in sorted(pairs):
        cell_keys = list(CELL_ORDER) + (list(NULL_CELLS) if pair_id in null_pairs else [])
        latents = {key: encode_cell(wm, *pairs[pair_id][key], device) for key in cell_keys}
        groups = {key: load_cell_groups(*pairs[pair_id][key]) for key in cell_keys}
        per_scene[pair_id] = evaluate_scene(wm, predictor, latents, groups, args.step, comps, args.n_probes, generator, args.blocks)
        print(f"[modulation] scene {pair_id} done ({time.time() - t0:.1f}s)", file=sys.stderr)
    agg = aggregate(per_scene, comps, args.n_boot, args.n_perm, rng, args.alpha)
    seeds = sorted({int(pairs[pid][CELL_ORDER[0]][1]["seed"]) for pid in pairs})
    out = {
        "tool": "modulation_patch (E2 action-entry path patching)",
        "model_label": args.model_label, "model_name": args.model_name,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint), "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts], "pair_ids": sorted(pairs), "seeds": seeds, "n_scenes": len(pairs),
        "null_factor_status": null_status, "n_null_scenes": len(null_pairs),
        "n_blocks": n_layers_of(predictor), "components": list(comps), "step": args.step, "n_probes": args.n_probes,
        "thresholds": {"recovery": 0.30, "null_recovery": 0.10, "cosine": 0.5, "alpha": args.alpha},
        **agg,
        "per_scene": per_scene,
        "interpretation_scope": ("Causal mediation on the AdaLN conditioning pathway (the only place the action exists in the predictor); "
                                 "latent-forecast endpoint on route tokens at imagined step 0. " + ZERO_SHOT_SCOPE
                                 + {"none": " No H0' cells.", "partial": f" H0' cells for {len(null_pairs)}/{len(pairs)} scenes.", "full": ""}[null_status]),
        "runtime_s": time.time() - t0, "torch_version": torch.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "modulation_patch.json").write_text(canonical_json(json_safe(out)) + "\n")
    rows = [{"key": k, "recovery": round(e["recovery"]["point"], 3), "null": (round(e["null_recovery"]["point"], 3) if e["null_recovery"]["point"] == e["null_recovery"]["point"] else None),
             "cosine": round(e["cosine"]["point"], 3), "p_maxt": e.get("recovery_minus_null_test", {}).get("p_maxt_fwer"), "E2": e["E2_single_block"]} for k, e in agg["table"].items()]
    print(json.dumps(json_safe({"n_scenes": len(pairs), "runtime_s": out["runtime_s"], "ceiling_route_recovery": agg["ceiling"]["route_recovery"]["point"],
                                "verdict": agg["verdict"], "table": rows}), indent=1))


if __name__ == "__main__":
    main()
