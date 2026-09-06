#!/usr/bin/env python3
"""Step 10: predictor-wide hazard x action interaction localization, per token group.

Othello-style layer sweep over every ``site = layer x hook`` of the JEPA-WM AdaLN
predictor and every imagined step, restricted to spatial token groups (egg,
gripper, corridor, background, gripper+corridor, all; see ``token_groups.py``).

Per scene (pair) ``i``, site, step and group we form the token-pooled effect
vectors (mean over the group's tokens, ``d``-dimensional):

- hazard main effect      ``mean_A (a[1,A] - a[0,A])``
- action main effect      ``mean_H (a[H,1] - a[H,0])``
- hazard x action DiD     ``I_i = a11 - a10 - a01 + a00``
- null-factor DiD         ``I'_i = (a[2,1] - a[2,0]) - (a[0,1] - a[0,0])`` when the
                          matched second off-path placement ``hazard == 2``
                          (``h2a0``/``h2a1``, from ``augment_null_factor.py``) exists
- relational interaction  ``I_i - I'_i``

Decision statistic (per site/step/group): leave-one-scene-out cross-validated
projection of ``I_i`` onto the unit mean of the other scenes (``cgs_stats``),
which has zero expectation under the null.  Significance: sign-flip
randomization over scenes, max-T across all sites within a (step, group) for
FWER, Benjamini-Yekutieli FDR over the descriptive map.  The token-level RMS
statistics are kept as descriptive only, with scene-clustered bootstrap CIs.

Partial null factor: when only a subset of scenes carries ``h2`` cells the
null-factor and relational statistics are computed on that subset
(``null_factor: partial``, ``n_null_scenes``); ``--require-null-factor`` refuses
only when no scene has them.  ``adaln`` sites are reported as the
action-availability reference only: their modulation vector is identical across
scenes for a given action, so their cross-scene sd is ~0 and any t is an
artifact; they are excluded from ranking, permutation and FDR.

Magnitude-aware reporting (arXiv:2602.07050 / review): a consistent direction
with negligible magnitude is the AdaLN element-wise artifact.  Per site/step/group
we therefore add ``ratio_to_action`` (interaction CV mean / |action main-effect CV
mean| at the same site/group), ``background_specificity`` (interaction CV mean /
|interaction CV mean at the background group of the same site/step|) and the
scene-level effect size ``d_z`` (mean/sd of the LOSO scores).  The explicit
ranking key is ``(magnitude_ok, t)`` with ``magnitude_ok = ratio >= --min-ratio and
specificity >= --min-specificity``; step-0 and step-1 maps are reported
separately and steps >= 1 are flagged ``autoregressive`` (the predicted history
can carry donor information).

Label rule: an interaction that is significant (max-T p < 0.05) at the ``egg``
group but not at ``gripper_corridor`` is tagged ``adaln_elementwise_candidate``
(hazard appearance modulated element-wise by the action pathway, not a
relational hazard x route representation).  The primary target is
``gripper_corridor``.

Activation dump (schema unchanged): ``<out>/activations/index.json`` and
``<out>/activations/<site_id>.npz`` with ``acts`` float16
``[n_cells, n_steps, n_tokens, d]``.  Rows cover every dumped cell in manifest
order, including the ``h2a*`` null-factor cells (index rows carry ``hazard = 2``)
so the H1-vs-H0' contrast can be formed downstream; ``--dump-hazard-levels``
overrides which hazard levels are written (default ``0 1 2``).  With
``--dump-groups`` only the union of the listed groups is stored (``n_tokens`` =
max group size, zero padded) together with ``token_index``
``[n_cells, n_steps, n_tokens]`` (-1 = pad).

``--domain driving`` (protocol v0.8, MetaDrive): token groups come from
``hazard_mask`` / ``corridor_mask`` (``token_groups.py``); the egg names remain
valid aliases (``egg`` -> ``hazard``, ``gripper_corridor`` -> ``corridor``,
``gripper`` -> empty) and the mapping is written to ``token_group_source``.
Hazard level 3 (cone on the same path) is dumped by default (``dump_hazard_levels``
0 1 2 3) and enters the inference as two extra contrasts per site/step/group:
``object`` = (a[3,1] - a[3,0]) - (a[0,1] - a[0,0]) and ``identity`` =
interaction - object (the identity x action contrast, ``protocol.did_by_identity``).
The primary group is ``hazard_corridor`` (alias ``gripper_corridor``).

References: arXiv:2309.16042, arXiv:2510.00845, arXiv:2606.27510,
arXiv:2511.04638, arXiv:2602.07050.
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

from cgs_stats import benjamini_yekutieli, cluster_bootstrap, cv_projection, json_safe, sign_flip_maxt  # noqa: E402
from model_action_sensitivity import encode_frames, load_model  # noqa: E402
from predictor_hooks import (  # noqa: E402
    HOOK_POINTS,
    ZERO_SHOT_SCOPE,
    PredictorRecorder,
    Site,
    all_sites,
    n_layers_of,
    predictor_of,
    site_order_key,
    spatial_tokens_of,
    unroll_from_latents,
)
from protocol import CELL_ORDER, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from token_groups import DOMAINS, DRIVING_ALIASES, PRIMARY_GROUP, CellGroups, alias_note, frame_for_step, group_names, load_cell_groups, set_domain, summarize, union  # noqa: E402


CALIBRATION_SEEDS = (101, 102)
NULL_CELLS = ((2, 0), (2, 1))
IDENTITY_CELLS = ((OBJECT_HAZARD, 0), (OBJECT_HAZARD, 1))
DEFAULT_GROUPS = ("gripper_corridor", "egg", "gripper", "corridor", "background", "all")
APPEARANCE_GROUP = {"egg": "egg", "driving": "hazard"}


def canon(name: str, domain: str) -> str:
    """Canonical (native) token-group name: resolves the egg-name aliases in the driving domain."""

    return DRIVING_ALIASES.get(name, name) if domain == "driving" else name


def identity_pairs(pairs: dict[str, dict[tuple[int, int], Any]]) -> list[str]:
    return sorted(pid for pid, cells in pairs.items() if set(IDENTITY_CELLS) <= set(cells))


def identity_status(pairs: dict[str, dict[tuple[int, int], Any]]) -> str:
    n = len(identity_pairs(pairs))
    return "none" if n == 0 else ("full" if n == len(pairs) else "partial")


# --------------------------------------------------------------------------- #
# Stimulus IO
# --------------------------------------------------------------------------- #


def read_manifest(artifacts: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (artifacts / "manifest.jsonl").read_text().splitlines() if line.strip()]


def merge_manifests(artifact_dirs: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for directory in artifact_dirs:
        if (directory / "manifest.jsonl").exists():
            rows.extend((directory, row) for row in read_manifest(directory))
        else:  # a parent holding seed_*/ subdirectories
            for sub in sorted(directory.glob("*/manifest.jsonl")):
                rows.extend((sub.parent, row) for row in read_manifest(sub.parent))
    return rows


def select_pairs(
    rows: list[tuple[Path, dict[str, Any]]],
    seeds_file: Path | None,
    allow_calibration: bool,
) -> dict[str, dict[tuple[int, int], tuple[Path, dict[str, Any]]]]:
    """Complete quartets (plus optional hazard==2 null cells) keyed by pair_id."""

    allowed: set[int] | None = None
    if seeds_file is not None:
        allowed = {int(tok) for tok in seeds_file.read_text().split() if tok.strip()}
    grouped: dict[str, dict[tuple[int, int], tuple[Path, dict[str, Any]]]] = {}
    for directory, row in rows:
        seed = int(row["seed"])
        if allowed is not None and seed not in allowed:
            continue
        if not allow_calibration and seed in CALIBRATION_SEEDS:
            continue
        grouped.setdefault(row["pair_id"], {})[(int(row["hazard"]), int(row["candidate_action"]))] = (directory, row)
    complete = {pid: cells for pid, cells in grouped.items() if set(CELL_ORDER) <= set(cells)}
    dropped = sorted(set(grouped) - set(complete))
    if dropped:
        print(f"[localize] dropping incomplete quartets: {dropped}", file=sys.stderr)
    return complete


def null_factor_pairs(pairs: dict[str, dict[tuple[int, int], Any]]) -> list[str]:
    return sorted(pid for pid, cells in pairs.items() if set(NULL_CELLS) <= set(cells))


def null_factor_status(pairs: dict[str, dict[tuple[int, int], Any]]) -> str:
    n = len(null_factor_pairs(pairs))
    return "none" if n == 0 else ("full" if n == len(pairs) else "partial")


def has_null_factor(pairs: dict[str, dict[tuple[int, int], Any]]) -> bool:
    return null_factor_status(pairs) == "full"


def magnitude_fields(
    entries: list[dict[str, Any]], min_ratio: float, min_specificity: float
) -> None:
    """Add ratio_to_action, background_specificity, d_z, magnitude_ok, autoregressive in place."""

    by_key = {(e["site_id"], e["step"], e["group"]): e for e in entries}
    for e in entries:
        cv = e["cv_projection"]
        inter = cv.get("interaction", {})
        mean = inter.get("mean")
        act = abs(cv.get("action", {}).get("mean") or 0.0)
        bg = by_key.get((e["site_id"], e["step"], "background"))
        bg_mean = abs((bg["cv_projection"].get("interaction", {}).get("mean") if bg else None) or 0.0)
        ratio = (mean / act) if (mean is not None and act > 1e-12) else None
        spec = (mean / bg_mean) if (mean is not None and bg_mean > 1e-12) else None
        if e["group"] == "background":
            spec = 1.0 if mean is not None else None
        e["ratio_to_action"] = ratio
        e["background_specificity"] = spec
        e["d_z"] = inter.get("d_z")
        e["magnitude_ok"] = bool(ratio is not None and spec is not None and ratio >= min_ratio and spec >= min_specificity)
        e["autoregressive"] = bool(int(e["step"]) >= 1)


def encode_cell(wm: Any, directory: Path, row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(directory / row["artifact"]) as cell:
        context = np.asarray(cell["context_frames"])
        future = np.asarray(cell["true_future_frames"])
        actions = np.asarray(cell["model_actions"])
    # One current frame seeds the unroll (matches the released planner and the
    # Phase-0 gate); ctxt_window only governs the rolling predicted history.
    z_context = encode_frames(wm, context[-1:], device)
    z_future = encode_frames(wm, future, device)
    act = torch.from_numpy(actions).to(device=device, dtype=torch.float32).unsqueeze(1)
    return {"z_context": z_context, "z_future": z_future, "actions": act}


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #


def rms(x: torch.Tensor | np.ndarray) -> float:
    if isinstance(x, torch.Tensor):
        return float(torch.sqrt(torch.mean(x.float() ** 2)).item())
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def quartet_effects(cells: dict[tuple[int, int], torch.Tensor]) -> dict[str, Any]:
    """Token-level main effects and interaction for one quartet of same-shape activations."""

    a00, a01, a10, a11 = (cells[key].float() for key in CELL_ORDER)
    hazard = 0.5 * ((a10 - a00) + (a11 - a01))
    action = 0.5 * ((a01 - a00) + (a11 - a10))
    interaction = a11 - a10 - a01 + a00
    mean = 0.25 * (a00 + a01 + a10 + a11)
    scale = torch.sqrt(torch.mean(torch.stack([(c - mean) ** 2 for c in (a00, a01, a10, a11)])))
    h, a, i = rms(hazard), rms(action), rms(interaction)
    scale_f = float(scale.item())
    per_token = torch.sqrt(torch.mean(interaction**2, dim=-1)).reshape(-1)
    return {
        "hazard_main_rms": h,
        "action_main_rms": a,
        "interaction_rms": i,
        "scale_rms": scale_f,
        "interaction_norm": i / max(scale_f, 1e-12),
        "interaction_share": i / max(h + a + i, 1e-12),
        "hazard_norm": h / max(scale_f, 1e-12),
        "action_norm": a / max(scale_f, 1e-12),
        "per_token_interaction_rms": per_token.cpu().numpy().astype(np.float32),
        "pooled": {
            "hazard": hazard.mean(dim=0).cpu().numpy().astype(np.float64),
            "action": action.mean(dim=0).cpu().numpy().astype(np.float64),
            "interaction": interaction.mean(dim=0).cpu().numpy().astype(np.float64),
        },
    }


def null_factor_vector(cells: dict[tuple[int, int], torch.Tensor]) -> np.ndarray:
    """Token-pooled (H0' - H0) x A interaction."""

    a00, a01 = cells[(0, 0)].float(), cells[(0, 1)].float()
    a20, a21 = cells[(2, 0)].float(), cells[(2, 1)].float()
    return ((a21 - a20) - (a01 - a00)).mean(dim=0).cpu().numpy().astype(np.float64)


def object_vector(cells: dict[tuple[int, int], torch.Tensor]) -> np.ndarray:
    """Token-pooled (H3 - H0) x A interaction (object on the path, v0.8)."""

    a00, a01 = cells[(0, 0)].float(), cells[(0, 1)].float()
    a30, a31 = cells[(OBJECT_HAZARD, 0)].float(), cells[(OBJECT_HAZARD, 1)].float()
    return ((a31 - a30) - (a01 - a00)).mean(dim=0).cpu().numpy().astype(np.float64)


def dump_keys(cell_keys: list[tuple[int, int]], levels: list[int]) -> list[tuple[int, int]]:
    """Cells written to the activation dump: those whose hazard level is in ``levels``."""

    return [key for key in cell_keys if key[0] in set(levels)]


def group_tokens_for_step(groups: dict[tuple[int, int], CellGroups], keys: list[tuple[int, int]], step: int, name: str, offset: int) -> np.ndarray:
    frame = frame_for_step(step, offset)
    return union(*(groups[k].group(frame, name) for k in keys))


def select_tokens(act: torch.Tensor, index: np.ndarray) -> torch.Tensor:
    """act [n_tokens, d] -> [len(index), d]; adaln sites (1 token) ignore the index."""

    if act.shape[0] == 1:
        return act
    return act[torch.as_tensor(index, dtype=torch.long, device=act.device)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True, help="stimulus dirs or parents of seed_*/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="jepa_wm_droid")
    parser.add_argument("--model-label", default="JEPA-WM DROID interaction localization")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hooks", nargs="+", default=list(HOOK_POINTS))
    parser.add_argument("--sites", nargs="*", default=None, help="explicit site ids (overrides --hooks/--max-layers)")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--no-embed-site", action="store_true")
    parser.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    parser.add_argument("--group-frame-offset", type=int, default=0, help="frame = step + offset for token groups")
    parser.add_argument("--dump-hooks", nargs="*", default=["resid_post"])
    parser.add_argument("--dump-groups", nargs="*", default=None, help="restrict dumped tokens to the union of these groups")
    parser.add_argument("--dump-hazard-levels", type=int, nargs="*", default=None, help="hazard levels written to the dump (default 0 1 2; driving 0 1 2 3)")
    parser.add_argument("--domain", choices=list(DOMAINS), default="egg", help="driving = v0.8 MetaDrive scenes (hazard/corridor groups, level 3 identity contrast)")
    parser.add_argument("--seeds-file", type=Path, default=None)
    parser.add_argument("--allow-calibration-seeds", action="store_true")
    parser.add_argument("--require-null-factor", action="store_true", help="refuse when NO scene has h2 cells")
    parser.add_argument("--reuse-dump", type=Path, default=None, help="existing <out>/activations dir to reuse (skips dumping)")
    parser.add_argument("--min-ratio", type=float, default=0.05)
    parser.add_argument("--min-specificity", type=float, default=2.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    domain = set_domain(args.domain)
    if args.dump_hazard_levels is None:
        args.dump_hazard_levels = [0, 1, 2, OBJECT_HAZARD] if domain == "driving" else [0, 1, 2]
    primary_group = PRIMARY_GROUP[domain]

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, args.allow_calibration_seeds)
    if not pairs:
        raise SystemExit("no complete quartets selected (calibration seeds are excluded by default)")
    null_status = null_factor_status(pairs)
    null_pairs = set(null_factor_pairs(pairs))
    null_present = null_status != "none"
    if args.require_null_factor and not null_present:
        raise SystemExit("--require-null-factor: hazard==2 (H0') cells are absent in every scene; refusing confirmatory output")
    ident_status = identity_status(pairs) if domain == "driving" else "none"
    ident_pairs = set(identity_pairs(pairs)) if domain == "driving" else set()
    pair_ids = sorted(pairs)
    keys_for = {pid: list(CELL_ORDER) + (list(NULL_CELLS) if pid in null_pairs else []) + (list(IDENTITY_CELLS) if pid in ident_pairs else []) for pid in pair_ids}
    n_cells = sum(len(dump_keys(keys_for[pid], args.dump_hazard_levels)) for pid in pair_ids)
    reuse_dump = args.reuse_dump is not None and (args.reuse_dump / "index.json").exists()
    if args.reuse_dump is not None and not reuse_dump:
        print(f"[localize] --reuse-dump {args.reuse_dump} has no index.json; dumping afresh", file=sys.stderr)

    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor = predictor_of(wm)
    n_layers = n_layers_of(predictor)
    n_spatial = spatial_tokens_of(wm)

    if args.sites:
        sites = [Site.parse(s) for s in args.sites]
    else:
        layers = n_layers if args.max_layers is None else min(n_layers, args.max_layers)
        sites = all_sites(layers, args.hooks, include_embed=not args.no_embed_site)
    dump_sites = set() if reuse_dump else {s.site_id for s in sites if s.hook in set(args.dump_hooks or [])}
    groups_wanted = [g for g in args.groups if g in group_names(domain, include_aliases=True)]

    # token groups for every cell up front (needed for the dump shape)
    cell_groups: dict[str, dict[tuple[int, int], CellGroups]] = {
        pid: {key: load_cell_groups(*pairs[pid][key]) for key in keys_for[pid]} for pid in pair_ids
    }
    group_sources = sorted({g.source for by in cell_groups.values() for g in by.values()})
    dump_k = n_spatial
    if args.dump_groups:
        dump_k = 0
        for pid in pair_ids:
            for key in keys_for[pid]:
                for frame in range(cell_groups[pid][key].n_frames):
                    dump_k = max(dump_k, len(union(*(cell_groups[pid][key].group(frame, g) for g in args.dump_groups))))

    act_dir = args.output / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    index_cells: list[dict[str, Any]] = []
    memmaps: dict[str, np.memmap] = {}
    token_index_mm: dict[str, np.memmap] = {}
    site_meta: dict[str, dict[str, Any]] = {}
    # site -> step -> group -> stat -> [{pair_id, value}]
    descriptive: dict[str, dict[int, dict[str, dict[str, list[dict[str, Any]]]]]] = {}
    # (step, group) -> kind -> site -> [n_scenes, d]
    vectors: dict[tuple[int, str], dict[str, dict[str, list[np.ndarray]]]] = {}
    token_maps: dict[str, dict[int, list[np.ndarray]]] = {}
    group_sizes: dict[str, dict[int, dict[str, list[int]]]] = {}
    n_steps_seen: int | None = None
    row_index = 0

    reference_vectors: dict[tuple[int, str], dict[str, dict[str, list[np.ndarray]]]] = {}  # adaln only
    null_scene_ids: list[str] = []
    for pair_id in pair_ids:
        cells = pairs[pair_id]
        groups = cell_groups[pair_id]
        cell_keys = keys_for[pair_id]
        pair_has_null = pair_id in null_pairs
        pair_has_ident = pair_id in ident_pairs
        if pair_has_null:
            null_scene_ids.append(pair_id)
        recorded: dict[tuple[int, int], dict[str, dict[int, torch.Tensor]]] = {}
        for key in cell_keys:
            directory, row = cells[key]
            latents = encode_cell(wm, directory, row, device)
            recorder = PredictorRecorder(predictor, sites, n_spatial, store_device=device, dtype=torch.float32)
            unroll_from_latents(wm, latents["z_context"], latents["actions"], recorder=recorder)
            if n_steps_seen is None:
                n_steps_seen = recorder.n_steps
            elif recorder.n_steps != n_steps_seen:
                raise RuntimeError(f"imagined-step count changed: {recorder.n_steps} vs {n_steps_seen}")
            recorded[key] = recorder.acts
            if key not in dump_keys(cell_keys, args.dump_hazard_levels):
                continue
            index_cells.append(
                {
                    "row": row_index,
                    "pair_id": pair_id,
                    "seed": int(row["seed"]),
                    "hazard": key[0],
                    "candidate_action": key[1],
                    "cell_id": row["cell_id"],
                    "artifact": str(directory / row["artifact"]),
                    "token_groups": summarize(groups[key]),
                }
            )
            for site in sites:
                sid = site.site_id
                acts = recorder.acts[sid]
                if sid not in site_meta:
                    sample = acts[0]
                    site_meta[sid] = {
                        "layer": site.layer,
                        "hook": site.hook,
                        "token_group": site.group_tokens,
                        "n_tokens": int(sample.shape[1]),
                        "d": int(sample.shape[2]),
                    }
                if sid in dump_sites:
                    k = dump_k if site.hook != "adaln" else 1
                    if sid not in memmaps:
                        shape = (n_cells, recorder.n_steps, k, site_meta[sid]["d"])
                        memmaps[sid] = np.lib.format.open_memmap(act_dir / f"{sid}.npy", mode="w+", dtype=np.float16, shape=shape)
                        if args.dump_groups:
                            token_index_mm[sid] = np.lib.format.open_memmap(
                                act_dir / f"{sid}.token_index.npy", mode="w+", dtype=np.int64, shape=(n_cells, recorder.n_steps, k)
                            )
                            token_index_mm[sid][:] = -1
                    for step, tensor in acts.items():
                        arr = tensor[0].to(torch.float16).cpu().numpy()
                        if args.dump_groups and site.hook != "adaln":
                            frame = frame_for_step(step, args.group_frame_offset)
                            idx = union(*(groups[key].group(frame, g) for g in args.dump_groups))
                            memmaps[sid][row_index, step, : len(idx)] = arr[idx]
                            token_index_mm[sid][row_index, step, : len(idx)] = idx
                        else:
                            memmaps[sid][row_index, step] = arr
            row_index += 1

        quartet_keys = list(CELL_ORDER)
        for site in sites:
            sid = site.site_id
            for step in range(n_steps_seen or 0):
                full = {key: recorded[key][sid][step][0] for key in cell_keys}
                eff_all = quartet_effects({k: full[k] for k in quartet_keys})
                token_maps.setdefault(sid, {}).setdefault(step, []).append(eff_all.pop("per_token_interaction_rms"))
                for gname in groups_wanted:
                    if site.hook == "adaln" and gname != "all":
                        continue  # the modulation vector has no spatial extent
                    idx = group_tokens_for_step(groups, cell_keys, step, gname, args.group_frame_offset)
                    group_sizes.setdefault(sid, {}).setdefault(step, {}).setdefault(gname, []).append(int(len(idx)))
                    if len(idx) == 0 and site.hook != "adaln":
                        continue
                    sub = {k: select_tokens(full[k], idx) for k in cell_keys}
                    eff = eff_all if gname == "all" else quartet_effects({k: sub[k] for k in quartet_keys})
                    pooled = eff["pooled"]
                    bucket = descriptive.setdefault(sid, {}).setdefault(step, {}).setdefault(gname, {})
                    for stat, value in eff.items():
                        if stat in ("pooled", "per_token_interaction_rms"):
                            continue
                        bucket.setdefault(stat, []).append({"pair_id": pair_id, "value": float(value)})
                    store = reference_vectors if site.hook == "adaln" else vectors
                    vec = store.setdefault((step, gname), {})
                    for kind in ("interaction", "hazard", "action"):
                        vec.setdefault(kind, {}).setdefault(sid, []).append(pooled[kind])
                    if pair_has_null:
                        nv = null_factor_vector(sub)
                        vec.setdefault("null_factor", {}).setdefault(sid, []).append(nv)
                        vec.setdefault("relational", {}).setdefault(sid, []).append(pooled["interaction"] - nv)
                    if pair_has_ident:
                        ov = object_vector(sub)
                        vec.setdefault("object", {}).setdefault(sid, []).append(ov)
                        vec.setdefault("identity", {}).setdefault(sid, []).append(pooled["interaction"] - ov)
        del recorded
        print(f"[localize] pair {pair_id} done ({time.time() - t0:.1f}s)", file=sys.stderr)

    # ---- finalize dumps (schema unchanged: acts [n_cells, n_steps, n_tokens, d]) ----
    dump_bytes = 0
    for sid, mm in memmaps.items():
        mm.flush()
        payload = {"acts": np.asarray(mm)}
        if sid in token_index_mm:
            token_index_mm[sid].flush()
            payload["token_index"] = np.asarray(token_index_mm[sid])
        np.savez(act_dir / f"{sid}.npz", **payload)
        (act_dir / f"{sid}.npy").unlink()
        if sid in token_index_mm:
            (act_dir / f"{sid}.token_index.npy").unlink()
        dump_bytes += (act_dir / f"{sid}.npz").stat().st_size
    if reuse_dump:
        try:
            act_dir.rmdir()
        except OSError:
            pass
    else:
      (act_dir / "index.json").write_text(
        json.dumps(
            {
                "cells": index_cells,
                "n_steps": n_steps_seen,
                "sites": {sid: {**site_meta[sid], "n_tokens": int(memmaps[sid].shape[2])} for sid in sorted(memmaps)},
                "dtype": "float16",
                "dump_groups": args.dump_groups,
                "dump_hazard_levels": args.dump_hazard_levels,
                "group_frame_offset": args.group_frame_offset,
                "layout": "acts[row, imagined_step, token, d]; rows = cells listed in 'cells' (hazard 0/1 quartet cells and, "
                "when present, hazard=2 null-factor cells); tokens = last-frame spatial tokens (or 1 AdaLN modulation "
                "vector); with dump_groups, token_index[row, step, token] gives the spatial token id (-1 = pad)",
                **({"domain": domain, "token_group_source": alias_note(domain), "identity_factor": ident_status} if domain != "egg" else {}),
            },
            indent=2,
        )
      )
    np.savez(
        args.output / "token_interaction_maps.npz",
        **{f"{sid}__s{step}": np.mean(np.stack(maps), axis=0) for sid, by_step in token_maps.items() for step, maps in by_step.items()},
    )

    # ---- inference ----
    site_results: list[dict[str, Any]] = []
    perm_summaries: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for (step, gname), kinds in sorted(vectors.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mats = {kind: {sid: np.stack(v) for sid, v in per_site.items()} for kind, per_site in kinds.items()}
        perm = {kind: sign_flip_maxt(mats[kind], args.n_perm, rng) for kind in mats}
        perm_summaries[f"s{step}__{gname}"] = {
            kind: {k: v for k, v in res.items() if k != "sites"} for kind, res in perm.items()
        }
        for sid in mats["interaction"]:
            site = Site.parse(sid)
            entry: dict[str, Any] = {
                "site_id": sid,
                "step": step,
                "group": gname,
                **site_meta[sid],
                "order_key": list(site_order_key(site, step)),
                "group_size_mean": float(np.mean(group_sizes[sid][step][gname])),
                "descriptive": {},
                "cv_projection": {},
            }
            for stat, rows in descriptive[sid][step][gname].items():
                values = np.asarray([r["value"] for r in rows])
                entry["descriptive"][stat] = cluster_bootstrap(values, args.n_boot, rng)
                entry["descriptive"][stat]["per_pair"] = {r["pair_id"]: r["value"] for r in rows}
            for kind in mats:
                proj = cv_projection(mats[kind][sid])
                proj.update(perm[kind]["sites"][sid])
                entry["cv_projection"][kind] = proj
            raw_p[f"{sid}|s{step}|{gname}"] = entry["cv_projection"]["interaction"]["p_raw"]
            site_results.append(entry)

    # adaln: action-availability reference only (no permutation, ranking or FDR)
    reference_results: list[dict[str, Any]] = []
    for (step, gname), kinds in sorted(reference_vectors.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for sid in kinds.get("interaction", {}):
            entry = {
                "site_id": sid, "step": step, "group": gname, **site_meta[sid], "role": "action_availability_reference",
                "cv_projection": {kind: cv_projection(np.stack(per_site[sid])) for kind, per_site in kinds.items() if sid in per_site},
                "descriptive": {},
            }
            for stat, rows in descriptive[sid][step][gname].items():
                values = np.asarray([r["value"] for r in rows])
                entry["descriptive"][stat] = cluster_bootstrap(values, args.n_boot, rng)
            reference_results.append(entry)

    fdr = benjamini_yekutieli(raw_p, args.alpha)
    for entry in site_results:
        key = f"{entry['site_id']}|s{entry['step']}|{entry['group']}"
        entry["cv_projection"]["interaction"]["q_by"] = fdr["q"].get(key)
        entry["labels"] = []
        entry["role"] = "candidate"
        p_fwer = entry["cv_projection"]["interaction"].get("p_maxt_fwer")
        entry["significant_maxt"] = bool(p_fwer is not None and not np.isnan(p_fwer) and p_fwer < args.alpha)
    magnitude_fields(site_results, args.min_ratio, args.min_specificity)

    # egg-only (driving: hazard-only) interactions -> adaln_elementwise_candidate
    by_key = {(e["site_id"], e["step"], e["group"]): e for e in site_results}
    by_canon = {(sid, step, canon(g, domain)): e for (sid, step, g), e in by_key.items()}
    for (sid, step, gname), entry in by_key.items():
        cg = canon(gname, domain)
        if cg == APPEARANCE_GROUP[domain] and entry["significant_maxt"]:
            gc = by_canon.get((sid, step, primary_group))
            if gc is None or not gc["significant_maxt"]:
                entry["labels"].append("adaln_elementwise_candidate")
        if cg == primary_group and entry["significant_maxt"]:
            entry["labels"].append("relational_candidate")

    def _t(e: dict[str, Any]) -> float:
        v = e["cv_projection"]["interaction"].get("t")
        return v if (v is not None and v == v) else -np.inf

    rank_key = lambda e: (not e["magnitude_ok"], -_t(e))  # noqa: E731  explicit: magnitude_ok first, then t
    ranked = sorted(site_results, key=rank_key)
    primary = [e for e in site_results if canon(e["group"], domain) == primary_group]
    live = sorted((e for e in primary if e["significant_maxt"] and e["magnitude_ok"]), key=lambda e: tuple(e["order_key"]))

    def brief(e: dict[str, Any]) -> dict[str, Any]:
        cv = e["cv_projection"]
        rel = cv.get("relational", {})
        return {
            "site_id": e["site_id"], "step": e["step"], "group": e["group"], "t": cv["interaction"].get("t"),
            "p_maxt_fwer": cv["interaction"].get("p_maxt_fwer"), "q_by": cv["interaction"].get("q_by"),
            "cv_mean": cv["interaction"].get("mean"), "ratio_to_action": e["ratio_to_action"],
            "background_specificity": e["background_specificity"], "d_z": e["d_z"], "magnitude_ok": e["magnitude_ok"],
            "autoregressive": e["autoregressive"], "labels": e["labels"],
            "relational_t": rel.get("t"), "relational_p_maxt_fwer": rel.get("p_maxt_fwer"), "relational_mean": rel.get("mean"),
            "null_factor_t": cv.get("null_factor", {}).get("t"), "n_null_scenes": rel.get("n_scenes"),
            **({"identity_t": cv.get("identity", {}).get("t"), "identity_p_maxt_fwer": cv.get("identity", {}).get("p_maxt_fwer"),
                "identity_mean": cv.get("identity", {}).get("mean"), "object_t": cv.get("object", {}).get("t")} if domain == "driving" else {}),
        }

    steps_seen = sorted({e["step"] for e in site_results})
    maps_by_step = {
        f"step{s}": {
            "autoregressive": s >= 1,
            "ranking_key": "(magnitude_ok, t desc); magnitude_ok = ratio_to_action >= min_ratio and background_specificity >= min_specificity",
            "top20": [brief(e) for e in sorted((e for e in site_results if e["step"] == s), key=rank_key)[:20]],
            "top10_gripper_corridor": [brief(e) for e in sorted((e for e in primary if e["step"] == s), key=rank_key)[:10]],
        }
        for s in steps_seen
    }

    result = {
        "model_label": args.model_label,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(args.config),
        "artifacts": [str(p) for p in args.artifacts],
        "pair_ids": pair_ids,
        "seeds": sorted({c["seed"] for c in index_cells}),
        "calibration_seeds_included": bool({c["seed"] for c in index_cells} & set(CALIBRATION_SEEDS)),
        "n_scenes": len(pair_ids),
        "n_sites": len(sites),
        "n_steps": n_steps_seen,
        "n_layers": n_layers,
        "n_spatial_tokens": n_spatial,
        "groups": groups_wanted,
        "group_frame_offset": args.group_frame_offset,
        "token_groups": "masks" if group_sources == ["masks"] else ("bbox_fallback" if group_sources == ["bbox_fallback"] else "mixed"),
        "null_factor_present": null_present,
        "null_factor": null_status,
        "n_null_scenes": len(null_scene_ids),
        "null_scene_pair_ids": null_scene_ids,
        "require_null_factor": args.require_null_factor,
        "min_ratio": args.min_ratio,
        "min_specificity": args.min_specificity,
        "dump_reused_from": str(args.reuse_dump) if reuse_dump else None,
        **({"domain": domain, "token_group_source": alias_note(domain), "primary_group": primary_group,
            "identity_factor": ident_status, "n_identity_scenes": len(ident_pairs), "identity_scene_pair_ids": sorted(ident_pairs)} if domain != "egg" else {}),
        "sites": [s.site_id for s in sites],
        "dumped_sites": sorted(memmaps),
        "dump_groups": args.dump_groups,
        "dump_hazard_levels": args.dump_hazard_levels,
        "dump_bytes": dump_bytes,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "alpha": args.alpha,
        "permutation": perm_summaries,
        "fdr_by": {k: v for k, v in fdr.items() if k != "q"},
        "site_results": site_results,
        "reference_results": reference_results,
        "ranking_key": "(magnitude_ok, t desc); adaln excluded",
        "ranked": [brief(e) for e in ranked[:40]],
        "maps_by_step": maps_by_step,
        "earliest_live_gripper_corridor": [brief(e) for e in live[:10]],
        "interpretation_scope": (
            "Descriptive/statistical localization of where hazard and candidate action interact in the predictor's "
            "own activations, per token group. Not a causal claim; site selection requires patch_site.py gates. "
            + ZERO_SHOT_SCOPE
            + (" Token groups are bbox fallbacks (no masks): gripper/corridor groups are empty." if "bbox_fallback" in group_sources else "")
            + {"none": " No H0' null-factor cells: relational interaction not computed.",
               "partial": f" H0' null-factor cells exist for {len(null_scene_ids)}/{len(pair_ids)} scenes: null-factor and relational statistics use that subset only.",
               "full": ""}[null_status]
            + " Steps >= 1 are autoregressive (predicted history may carry donor information); step 0 is the clean map."
            + ({"none": " No level-3 (object on path) cells: identity contrast not computed.",
                "partial": f" Level-3 cells exist for {len(ident_pairs)}/{len(pair_ids)} scenes: object/identity statistics use that subset.",
                "full": " Identity contrast (interaction minus object DiD) computed on every scene."}[ident_status] if domain == "driving" else "")
        ),
        "runtime_s": time.time() - t0,
        "torch_version": torch.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "interaction_map.json").write_text(canonical_json(json_safe(result)) + "\n")
    print(json.dumps(json_safe({"n_scenes": len(pair_ids), "n_sites": len(sites), "n_steps": n_steps_seen, "token_groups": result["token_groups"],
                      "null_factor": null_status, "n_null_scenes": len(null_scene_ids), "dump_bytes": dump_bytes,
                      "top10_step0_gripper_corridor": maps_by_step.get("step0", {}).get("top10_gripper_corridor", [])}), indent=2))


if __name__ == "__main__":
    main()
