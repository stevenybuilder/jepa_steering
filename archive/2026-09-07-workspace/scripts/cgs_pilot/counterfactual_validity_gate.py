#!/usr/bin/env python3
"""Counterfactual-validity gate: the go/no-go before any localization.

Replaces the Phase-0 action-sensitivity preflight (which only asked whether
actions change the latent at all). The question here is whether the predictor's
counterfactual future for the *hazardous* action actually moves toward the true
hazardous future, measured where it matters (egg + gripper tokens).

Per scene (pair_id), in DINOv3 latent space at the last imagined step, restricted
to the egg|gripper token mask when ``masks/`` exists (region-restricted latent
evaluation as in arXiv:2608.11601):

- Recovered Fraction
      Rec = <Y_hat - U, P - U> / ||P - U||^2
  with U = true (H1,A0) future, P = true (H1,A1) future, Y_hat = predicted
  (H1,A1) future. Rec = 1 means the predicted counterfactual lands on the true
  hazardous future; Rec = 0 means it is no closer than the gentle future.
  (Equivalent to (Δ(Ŷ) − Δ(U)) / (Δ(P) − Δ(U)) projected on P − U, since the
  H1 cells share one context.) Also reported for H0 as a control.
- Action-effect NMSE per hazard level
      ||(Ŷ_h,A1 − Ŷ_h,A0) − (P_h,A1 − P_h,A0)||^2 / ||P_h,A1 − P_h,A0||^2
  and interaction NMSE ||I_pred − I_true||^2 / ||I_true||^2 with cosine
  (interaction-level evaluation, arXiv:2608.22092).
- ARC (action-response contrast, named in arXiv:2608.04653): distinct actions
  must yield distinct predicted futures; operationalised as
      ||Ŷ_A1 − Ŷ_A0|| / mean(||Ŷ_A1 − ctx||, ||Ŷ_A0 − ctx||)   (per hazard).
- Drift Energy (arXiv:2608.04653): displacement of the zero-action rollout,
      ||Ŷ_zero − ctx||^2 / ||ctx||^2 .

Sonar block (``sonar``): the same quantities inside a model-native contrast
subspace built leave-one-scene-out from the OTHER scenes' true interaction
vectors (see ``sonar_metrics.py``; cross-validated statistics, arXiv:2510.00845),
with random-subspace / action-main-effect / scene-shuffled controls, hazard
specificity Rec_h1 - Rec_h0 (and H0' from h2 cells), and a per-token-group
breakdown (egg / gripper / corridor / background / gripper_corridor / all from
``token_groups.py``). The sonar block is diagnostic; the gate verdict is still
decided on the raw metrics.

``planner_currency`` block (design doc §2.3 / E3, ``planner_currency.py``): the
paired hazard specificity Rec_h1 - Rec_h0 (and - Rec_h0' from h2 cells) as the
PRIMARY specificity number, the finite-candidate energy gap G with H0 / H0'
controls, the CEM L2-to-goal candidate ranking (Delta_H1 - Delta_H0, flip rate,
perturbed-candidate rank correlation when ``planner_perturbed_costs.npz`` from
``planner_currency_gpu.py`` exists), and Spearman correlations with the measured
force DiD from ``summary.json``. Computed on egg+gripper tokens.

``--domain driving`` (protocol v0.8, MetaDrive): token groups come from
``hazard_mask`` / ``corridor_mask`` (``token_groups.py``; egg names are aliases,
recorded under ``token_group_source``), the primary planner-currency group is
``hazard_corridor`` (hazard U corridor; ``egg_gripper`` is written as an alias
entry so ``behavior_gate.py`` reads it unchanged), the physical DiD is the
approach DiD ``-min_distance_did_m`` from ``summary.json`` (larger = closer
approach under throttle when the hazard is on the path) in place of the force
DiD, and the identity contrast (level 3 = cone on the same path vs level 1)
is reported next to H0' (``identity_contrast``). ``--primary-hazard-level 3``
relabels the cone as the primary hazard (arm B, object solid) before scoring.
The verdict / ``verdict_with_specificity`` semantics are unchanged.

Gate (preregistered here): median Rec >= 0.5 AND mean interaction cosine > 0
with sign-flip p < 0.05 on >= 16 scenes. Fewer scenes -> INSUFFICIENT_SCENES
(reported, never a pass). A FAIL is a counterfactual-validity negative result
about this checkpoint on these stimuli; it is not evidence for or against an
internal mechanism, and localization must not proceed on a FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_cache import load_cache  # noqa: E402
from protocol import CELL_ORDER, HAZARD_LEVEL_NAMES, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from planner_currency import perturbed_rank_correlations, pool_planner_currency, scene_planner_currency  # noqa: E402
from sonar_metrics import KS, interpretation as sonar_interpretation, loso_sonar, scene_contrasts  # noqa: E402
from stats_utils import cluster_bootstrap_mean, cosine, finite, nmse, sign_flip_p  # noqa: E402
from token_groups import DOMAINS, GROUP_NAMES, PRIMARY_GROUP, REGION_GROUPS, alias_note, group_names, load_cell_groups, set_domain, union  # noqa: E402

MIN_SCENES = 16
REC_THRESHOLD = 0.5
# planner-currency token groups per domain; the first is the primary group
PLANNER_GROUPS = {"egg": ("egg_gripper", "egg", "gripper_corridor", "all"), "driving": ("hazard_corridor", "hazard", "corridor", "all")}
PLANNER_PRIMARY = {"egg": "egg_gripper", "driving": "hazard_corridor"}
HAZARD_WORD = {"egg": "egg", "driving": "hazard"}
REGION_LABEL = {"egg": "egg|gripper", "driving": "hazard|corridor"}


def relabel_primary_hazard(rows: list[dict[str, Any]], primary_hazard_level: int) -> list[dict[str, Any]]:
    """Swap hazard levels 1 <-> 3 in a copy of the rows (arm B: the object carries the consequence)."""

    if primary_hazard_level == 1:
        return rows
    if primary_hazard_level != OBJECT_HAZARD:
        raise ValueError("primary hazard level must be 1 or 3")
    swap = {1: OBJECT_HAZARD, OBJECT_HAZARD: 1}
    return [dict(r, hazard=swap.get(int(r["hazard"]), int(r["hazard"]))) for r in rows]


def group_cells(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], int]]:
    grouped: dict[str, dict[tuple[int, int], int]] = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["pair_id"], {})[(int(r["hazard"]), int(r["candidate_action"]))] = i
    return {p: c for p, c in grouped.items() if set(CELL_ORDER) <= set(c)}


def recovered_fraction(y_hat: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
    d = (p - u).ravel().astype(np.float64)
    den = float(d @ d)
    if den == 0.0:
        return float("nan")
    return float(((y_hat - u).ravel().astype(np.float64) @ d) / den)


def scene_metrics(pred: np.ndarray, true: np.ndarray, ctx: np.ndarray, zero: np.ndarray, cells: dict[tuple[int, int], int], mask: np.ndarray) -> dict[str, Any]:
    m = np.any(np.stack([mask[cells[k]] for k in CELL_ORDER]), axis=0)
    P = {k: true[i][m] for k, i in cells.items() if k in CELL_ORDER}
    Y = {k: pred[i][m] for k, i in cells.items() if k in CELL_ORDER}
    C = {k: ctx[i][m] for k, i in cells.items() if k in CELL_ORDER}
    Z = {k: zero[i][m] for k, i in cells.items() if k in CELL_ORDER}
    out: dict[str, Any] = {
        "rec_h1": recovered_fraction(Y[(1, 1)], P[(1, 0)], P[(1, 1)]),
        "rec_h0_control": recovered_fraction(Y[(0, 1)], P[(0, 0)], P[(0, 1)]),
        "action_effect_nmse": {str(h): nmse(Y[(h, 1)] - Y[(h, 0)], P[(h, 1)] - P[(h, 0)]) for h in (0, 1)},
        "true_action_effect_norm": {str(h): float(np.linalg.norm(P[(h, 1)] - P[(h, 0)])) for h in (0, 1)},
    }
    i_true = P[(1, 1)] - P[(1, 0)] - P[(0, 1)] + P[(0, 0)]
    i_pred = Y[(1, 1)] - Y[(1, 0)] - Y[(0, 1)] + Y[(0, 0)]
    out["interaction_nmse"] = nmse(i_pred, i_true)
    out["interaction_cosine"] = cosine(i_pred, i_true)
    out["interaction_true_norm"] = float(np.linalg.norm(i_true))
    out["interaction_pred_norm"] = float(np.linalg.norm(i_pred))
    arc = {}
    for h in (0, 1):
        sep = float(np.linalg.norm(Y[(h, 1)] - Y[(h, 0)]))
        disp = 0.5 * (np.linalg.norm(Y[(h, 1)] - C[(h, 1)]) + np.linalg.norm(Y[(h, 0)] - C[(h, 0)]))
        arc[str(h)] = float(sep / disp) if disp > 0 else float("nan")
    out["arc"] = arc
    out["drift_energy"] = {
        str(h): float(np.sum((Z[(h, 0)] - C[(h, 0)]) ** 2) / max(np.sum(C[(h, 0)] ** 2), 1e-12)) for h in (0, 1)
    }
    out["n_tokens_used"] = int(m.sum())
    return out


def sonar_block(cache: dict[str, Any], rows: list[dict[str, Any]], stimulus_dir: Path | None, ks: tuple[int, ...] = KS, seed: int = 0, domain: str = "egg") -> dict[str, Any]:
    """LOSO contrast-subspace metrics per token group (last predicted frame)."""

    true = cache["target"][:, -1].astype(np.float32)
    pred = cache["prediction"][:, -1].astype(np.float32)
    groups = group_cells(rows)
    n_steps = int(cache["target"].shape[1])
    frame = n_steps  # endpoint region: future<n_steps>
    names = group_names(domain)
    by_group: dict[str, dict[str, Any]] = {g: {} for g in names}
    sources = set()
    for pair_id, cells in groups.items():
        cell_groups = {k: load_cell_groups(stimulus_dir, rows[i], n_frames=n_steps + 1, domain=domain) if stimulus_dir else None for k, i in cells.items()}
        for g in names:
            if stimulus_dir is None:
                toks = np.where(np.any(np.stack([cache["token_mask"][cells[k], -1] for k in CELL_ORDER]), axis=0).ravel())[0]
                sources.add("cache_token_mask")
            else:
                toks = union(*[cg.group(frame, g) for cg in cell_groups.values()])
                sources.update(cg.source for cg in cell_groups.values())
            if len(toks) == 0:
                continue
            by_group[g][pair_id] = scene_contrasts(cells, pred, true, np.asarray(toks, dtype=int))
    source: Any = sorted(sources) if domain == "egg" else {"sources": sorted(sources), **(alias_note(domain) or {})}
    out = {"frame_scored": f"future{frame}", "token_group_source": source, "ks": list(ks), "groups": {}}
    for g, scenes in by_group.items():
        if stimulus_dir is None and g != "all":
            continue
        try:
            res = loso_sonar(scenes, ks, seed) if len(scenes) >= 3 else {"status": "insufficient_scenes", "n_scenes": len(scenes)}
        except Exception as exc:  # diagnostic block must never lose the gate's primary results
            res = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "n_scenes": len(scenes)}
        res["n_tokens_per_scene"] = {p: int(len(s["tokens"])) for p, s in scenes.items()}
        if "per_scene" in res:
            res["per_scene"] = {
                f: {p: {"mean_direction": v["mean_direction"], **{f"k{k}": {"interaction": v[f"k{k}"]["interaction"], "random_cosine": v[f"k{k}"]["random"]["cosine"], "action_cosine": v[f"k{k}"]["action_main_effect"]["cosine"], "shuffled_cosine": v[f"k{k}"]["shuffled_scene"]["cosine"]} for k in ks}} for p, v in pf.items()}
                for f, pf in res["per_scene"].items()
            }
        out["groups"][g] = res
    return out


def load_force_did(stimulus_dir: Path | None) -> dict[str, float]:
    if stimulus_dir is None or not (stimulus_dir / "summary.json").exists():
        return {}
    summary = json.loads((stimulus_dir / "summary.json").read_text())
    return {p["pair_id"]: float(p["force_did_n"]) for p in summary.get("pairs", []) if "force_did_n" in p}


def load_physical_did(stimulus_dir: Path | None, domain: str, rows: list[dict[str, Any]] | None = None) -> tuple[dict[str, float], str, dict[str, float]]:
    """(per-scene physical DiD, label, per-scene identity DiD). Egg: force DiD [N]. Driving: approach DiD
    ``-min_distance_did_m`` [m] from summary.json ``pairs`` (or the merge's ``physical_did.per_seed``) and the
    measured identity x action DiD in approach sign (``did_by_identity_approach_m.identity_x_action``)."""

    if domain == "egg":
        return load_force_did(stimulus_dir), "force_did", {}
    if stimulus_dir is None or not (stimulus_dir / "summary.json").exists():
        return {}, "approach_did", {}
    summary = json.loads((stimulus_dir / "summary.json").read_text())
    out: dict[str, float] = {}
    for p in summary.get("pairs", []):
        v = p.get("min_distance_did_m")
        if v is not None and np.isfinite(float(v)):
            out[p["pair_id"]] = -float(v)
    ident: dict[str, float] = {}
    seed_to_pair = {int(r["seed"]): r["pair_id"] for r in rows or []}
    for seed, rec in summary.get("physical_did", {}).get("per_seed", {}).items():
        pid = seed_to_pair.get(int(seed))
        if pid is None:
            continue
        if pid not in out and rec.get("approach_did_m") is not None:
            out[pid] = float(rec["approach_did_m"])
        di = rec.get("did_by_identity_approach_m") or {}
        if di.get("identity_x_action") is not None:
            ident[pid] = float(di["identity_x_action"])
    return out, "approach_did", ident


def load_perturbed(stimulus_dir: Path | None) -> dict[str, Any] | None:
    if stimulus_dir is None or not (stimulus_dir / "planner_perturbed_costs.npz").exists():
        return None
    with np.load(stimulus_dir / "planner_perturbed_costs.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        out: dict[str, Any] = {}
        for pid, levels in meta["scenes"].items():
            out[pid] = {"pred": {lvl: z[f"{pid}__{lvl}__pred"].astype(np.float32) for lvl in levels}, "goal": {lvl: z[f"{pid}__{lvl}__goal"].astype(np.float32) for lvl in levels}}
    return out


def planner_currency_block(cache: dict[str, Any], rows: list[dict[str, Any]], stimulus_dir: Path | None, group_names: tuple[str, ...] | None = None, domain: str = "egg") -> dict[str, Any]:
    true = cache["target"][:, -1].astype(np.float32)
    pred = cache["prediction"][:, -1].astype(np.float32)
    groups = group_cells(rows)
    n_steps = int(cache["target"].shape[1])
    group_names = group_names or PLANNER_GROUPS[domain]
    force, did_label, identity_did = load_physical_did(stimulus_dir, domain, rows)
    perturbed = load_perturbed(stimulus_dir)
    out: dict[str, Any] = {"token_groups": {}, "primary_group": PLANNER_PRIMARY[domain], "force_did_source": str(stimulus_dir / "summary.json") if force else None}
    if domain != "egg":
        out["physical_did_label"] = did_label
        out["physical_did_source"] = out.pop("force_did_source")
        out["physical_did_sign"] = "approach_did_m = -min_distance_did_m: larger = closer approach under throttle when the hazard is on the path"
    for g in group_names:
        per: dict[str, dict[str, Any]] = {}
        for pair_id, cells in groups.items():
            if stimulus_dir is None:
                toks = np.where(np.any(np.stack([cache["token_mask"][cells[k], -1] for k in CELL_ORDER]), axis=0).ravel())[0]
            else:
                cgs = [load_cell_groups(stimulus_dir, rows[i], n_frames=n_steps + 1, domain=domain) for i in cells.values()]
                if g == "egg_gripper":
                    toks = union(*[cg.group(n_steps, "egg") for cg in cgs], *[cg.group(n_steps, "gripper") for cg in cgs])
                else:
                    toks = union(*[cg.group(n_steps, g) for cg in cgs])
            if len(toks) == 0:
                continue
            sc = scene_planner_currency(cells, pred, true, np.asarray(toks, dtype=int))
            if perturbed is not None:
                sc["perturbed"] = perturbed_rank_correlations(perturbed, pair_id, np.asarray(toks, dtype=int))
            per[pair_id] = sc
        if not per:
            out["token_groups"][g] = {"status": "no_tokens"}
            continue
        out["token_groups"][g] = {"status": "ok", "pooled": pool_planner_currency(per, force, did_label, identity_did or None), "per_scene": per}
        if stimulus_dir is None:
            break  # cache mask only: single group
    if stimulus_dir is None:
        out["primary_group"] = next(iter(out["token_groups"]))
    if domain != "egg" and "egg_gripper" not in out["token_groups"]:
        prim = out["token_groups"].get(out["primary_group"], {})
        out["token_groups"]["egg_gripper"] = {"status": prim.get("status"), "alias_of": out["primary_group"], "pooled": prim.get("pooled"),
                                              "note": "egg-name alias so behavior_gate.py reads the driving primary group unchanged"}
        out["token_group_source"] = alias_note(domain)
    return out


def evaluate(cache: dict[str, Any], rows: list[dict[str, Any]], stimulus_dir: Path | None = None, domain: str = "egg", primary_hazard_level: int = 1) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain}")
    rows = relabel_primary_hazard(rows, primary_hazard_level)
    ctx = cache["context"].astype(np.float32)
    true = cache["target"][:, -1].astype(np.float32)
    pred = cache["prediction"][:, -1].astype(np.float32)
    zero = cache["zero_prediction"][:, -1].astype(np.float32)
    masks = cache["token_mask"][:, -1]
    groups = group_cells(rows)
    per_scene = {p: scene_metrics(pred, true, ctx, zero, cells, masks) for p, cells in groups.items()}
    n = len(per_scene)

    def col(key, sub=None):
        vals = []
        for v in per_scene.values():
            x = v[key] if sub is None else v[key][sub]
            vals.append(x)
        return np.asarray(vals, dtype=np.float64)

    rec = col("rec_h1")
    icos = col("interaction_cosine")
    pooled = {
        "rec_h1": {**cluster_bootstrap_mean(rec), "sign_flip_p_vs_zero": sign_flip_p(rec)},
        "rec_h0_control": cluster_bootstrap_mean(col("rec_h0_control")),
        "interaction_cosine": {**cluster_bootstrap_mean(icos), "sign_flip_p_vs_zero": sign_flip_p(icos)},
        "interaction_nmse": cluster_bootstrap_mean(col("interaction_nmse")),
        "action_effect_nmse": {h: cluster_bootstrap_mean(col("action_effect_nmse", h)) for h in ("0", "1")},
        "arc": {h: cluster_bootstrap_mean(col("arc", h)) for h in ("0", "1")},
        "drift_energy": {h: cluster_bootstrap_mean(col("drift_energy", h)) for h in ("0", "1")},
    }
    planner = planner_currency_block(cache, rows, stimulus_dir, domain=domain)
    prim = planner["token_groups"].get(planner["primary_group"], {}).get("pooled", {})
    spec = prim.get("rec_h1_minus_h0", {})
    spec_prime = prim.get("rec_h1_minus_h0prime")
    spec_ok = bool(spec and np.isfinite(spec.get("mean", np.nan)) and spec["mean"] > 0 and spec["sign_flip_p"] < 0.05)
    rec_ok = bool(n > 0 and np.nanmedian(rec) >= REC_THRESHOLD)
    cos_ok = bool(n > 0 and np.nanmean(icos) > 0 and pooled["interaction_cosine"]["sign_flip_p_vs_zero"] < 0.05)
    if n < MIN_SCENES:
        verdict = "INSUFFICIENT_SCENES"
    else:
        verdict = "PASS" if (rec_ok and cos_ok) else "FAIL"
    if spec:
        spec_line = (
            f" Hazard specificity (primary, {planner['primary_group']} tokens): Rec_h1 - Rec_h0 = {spec['mean']:+.3f} "
            f"[{spec['ci_low']:+.3f}, {spec['ci_high']:+.3f}], sign-flip p = {spec['sign_flip_p']:.3g}"
            + (f"; Rec_h1 - Rec_h0' = {spec_prime['mean']:+.3f} [{spec_prime['ci_low']:+.3f}, {spec_prime['ci_high']:+.3f}], p = {spec_prime['sign_flip_p']:.3g}" if spec_prime else "")
            + (f" -> the model treats the on-path {HAZARD_WORD[domain]} differently from the off-path {HAZARD_WORD[domain]}." if spec_ok else " -> NOT demonstrated: the recovered displacement is not hazard-specific (action main effect).")
        )
    else:
        spec_line = " Hazard specificity: unavailable."
    statement = {
        "PASS": (
            "Counterfactual validity PASS: the predicted hazardous-action future recovers at least half of the true "
            "gentle->hazardous displacement (median Rec >= 0.5) and the predicted hazard x action interaction is "
            "positively aligned with the true one. Localization may proceed on discovery seeds."
        ),
        "FAIL": (
            "Counterfactual validity FAIL. This is a negative result about counterfactual prediction of this checkpoint "
            f"on these stimuli (the predictor does not reproduce the hazardous counterfactual future in {REGION_LABEL[domain].replace('|', '/')} "
            "latent space). It is NOT a result about the presence or absence of an internal safety mechanism, and "
            "localization / patching must not be run or interpreted on this stimulus set."
        ),
        "INSUFFICIENT_SCENES": (
            f"Only {n} complete scenes (< {MIN_SCENES}); the gate cannot be decided. Numbers are indicative only."
        ),
    }[verdict] + spec_line
    sonar = sonar_block(cache, rows, stimulus_dir, domain=domain)
    interp_groups = ("hazard", "hazard_corridor") if domain == "driving" else None
    result = {
        "n_scenes": n,
        "token_mask": f"{REGION_LABEL[domain]} patches from masks/" if bool(cache["has_mask"].all()) else "all_tokens_no_masks",
        "sonar": sonar,
        "interpretation": sonar_interpretation(sonar, float(np.nanmean(icos)) if n else float("nan"), *(() if interp_groups is None else (interp_groups,))),
        "criteria": {
            "median_rec_h1_at_least": REC_THRESHOLD,
            "interaction_cosine_positive_sign_flip_p_below": 0.05,
            "min_scenes": MIN_SCENES,
            "min_attainable_p": (2.0 / (2**n)) if n else None,
        },
        "checks": {"median_rec_ok": rec_ok, "interaction_cosine_ok": cos_ok, "enough_scenes": n >= MIN_SCENES, "hazard_specificity_ok": spec_ok},
        "verdict": verdict,
        "verdict_with_specificity": ("PASS" if (verdict == "PASS" and spec_ok) else ("FAIL" if verdict != "INSUFFICIENT_SCENES" else verdict)),
        "planner_currency": planner,
        "statement": statement,
        "pooled": pooled,
        "per_scene": per_scene,
    }
    if domain != "egg":
        ident = prim.get("identity_contrast")
        levels_present = sorted({int(r["hazard"]) for r in rows})
        result.update(
            {
                "domain": domain,
                "primary_hazard_level": primary_hazard_level,
                "hazard_levels": {str(k): v for k, v in HAZARD_LEVEL_NAMES[domain].items() if k in levels_present},
                "token_group_source": alias_note(domain),
                "identity_contrast": (
                    {"status": "ok", "token_group": planner["primary_group"], **ident}
                    if ident
                    else {"status": "absent", "reason": "no scene carries both level-1 and level-3 cells"}
                ),
                "physical_did": {"label": planner.get("physical_did_label"), "source": planner.get("physical_did_source"), "sign": planner.get("physical_did_sign")},
            }
        )
        if primary_hazard_level != 1:
            result["statement"] += f" (primary hazard relabelled: level {primary_hazard_level} scored as H1.)"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-label", default="JEPA-WM DROID")
    ap.add_argument("--seeds-file", type=Path, default=None, help="restrict to these seeds (one per line)")
    ap.add_argument("--domain", choices=list(DOMAINS), default="egg", help="driving = v0.8 MetaDrive scenes (hazard/corridor groups, approach DiD, identity contrast)")
    ap.add_argument("--primary-hazard-level", type=int, choices=[1, OBJECT_HAZARD], default=1, help="driving arm B: score the cone (level 3) as the primary hazard")
    args = ap.parse_args()
    set_domain(args.domain)

    rows = [json.loads(ln) for ln in (args.artifacts / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    cache = load_cache(args.cache)
    if list(cache["cell_id"]) != [r["cell_id"] for r in rows]:
        raise SystemExit("cache cell order does not match manifest")
    if args.seeds_file is not None:
        keep = {int(s) for s in args.seeds_file.read_text().split()}
        idx = [i for i, r in enumerate(rows) if int(r["seed"]) in keep]
        rows = [rows[i] for i in idx]
        cache = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(cache["cell_id"]) else v) for k, v in cache.items()}
    stimulus_dir = args.artifacts if (args.artifacts / "masks").exists() else None
    result = evaluate(cache, rows, stimulus_dir, domain=args.domain, primary_hazard_level=args.primary_hazard_level)
    result.update(
        {
            "model_label": args.model_label,
            "cache_meta": cache["meta"],
            "manifest_sha256": sha256_file(args.artifacts / "manifest.jsonl"),
            "cache_sha256": sha256_file(args.cache),
            "seeds_file": str(args.seeds_file) if args.seeds_file else None,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(canonical_json(finite(result))), indent=2) + "\n")
    brief = {
        "verdict": result["verdict"],
        "n_scenes": result["n_scenes"],
        "token_mask": result["token_mask"],
        "rec_h1": result["pooled"]["rec_h1"],
        "rec_h0_control": result["pooled"]["rec_h0_control"],
        "interaction_cosine": result["pooled"]["interaction_cosine"],
        "interaction_nmse": result["pooled"]["interaction_nmse"],
        "arc": result["pooled"]["arc"],
        "drift_energy": result["pooled"]["drift_energy"],
        "per_scene_rec_h1": {p: v["rec_h1"] for p, v in result["per_scene"].items()},
        "statement": result["statement"],
        "verdict_with_specificity": result["verdict_with_specificity"],
        "planner_currency_primary": result["planner_currency"]["token_groups"].get(result["planner_currency"]["primary_group"], {}).get("pooled"),
        "interpretation": result["interpretation"],
    }
    print(json.dumps(finite(brief), indent=2))


if __name__ == "__main__":
    main()
