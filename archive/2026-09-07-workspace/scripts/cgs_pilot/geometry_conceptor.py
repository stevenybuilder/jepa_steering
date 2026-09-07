#!/usr/bin/env python3
"""Soft-conceptor geometry entrant (COAST, Miao et al. arXiv:2605.17144) over a predictor activation dump.

Per site x imagined step x token group:

1. Pool the group's tokens per cell (and keep matched per-token rows), residualize
   frozen nuisance (pixel signature, egg centroid / token position, proprio,
   action magnitude) with a scene-fold ridge, and form the activation-space DiD
   per scene ``delta_i = (h11 - h10) - (h01 - h00)`` plus the H-averaged action
   vector ``a_i = 0.5[(h01 - h00) + (h11 - h10)]``.
2. Fit ``C_int`` on the deltas and ``C_action`` on the action vectors (the generic
   motor geometry; ``C_null`` from H0' cells is added when a dump carries hazard-2
   cells), construct ``C_safety = C_int AND NOT C_action`` (AND NOT C_null when
   available), and the controls: matched-spectrum random conceptor and the
   rank-one mean-difference projector.  Aperture sweep alpha in {1,3,10,30,100}
   on rows scaled to unit mean squared norm, plus the aperture that hits a target
   quota (the frozen, label-free selection rule).
3. Score like the other tournament entrants: leave-one-scene-out refits of the
   conceptor and a ridge readout of contact / force / interaction from ``h C``;
   incremental value over the nuisance covariates; stability across three
   discovery refits; quota per site/step; conceptor similarity across imagined
   steps (does the subspace rotate) and adjacent layers (band structure); overlap
   between ``C_int`` and ``C_action``.
4. Export factored conceptors for ``patch_site.py`` (format in ``write_conceptor_npz``).

Descriptive only.  Multiplicative steering ``h' = h((1-b)I + bC)`` or suppression
``h' = h(I - bC)`` is the causal test and lives in ``patch_site.py``.
References: arXiv:2605.17144 (COAST), arXiv:2605.05115 (manifold steering),
arXiv:2602.07050 (circular population geometry).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_localize import cell_artifact, group_token_sets, interaction_labels, load_index, pool_site, quartets  # noqa: E402
from geometry_models import (  # noqa: E402
    alpha_for_quota,
    auroc,
    cluster_bootstrap,
    conceptor_and_not,
    conceptor_apply,
    conceptor_factored,
    conceptor_quota,
    conceptor_similarity,
    matched_spectrum_random,
    quartet_residualize,
    r_squared,
    rank_one_projector,
    ridge_fit,
    ridge_predict,
    scene_center,
    within_scene_did_rows,
)

ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0)
CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
TARGETS = {"contact": "binary", "force": "continuous", "interaction": "binary"}
GATES = {"auroc": 0.75, "r2": 0.30}


# --------------------------------------------------------------------------- #
# Nuisance covariates (pixel signature, egg centroid / token position, proprio, action magnitude)
# --------------------------------------------------------------------------- #


def pixel_signature(frame: np.ndarray, grid: int = 8) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float64) / 255.0
    H, W, C = f.shape
    return f[: H - H % grid, : W - W % grid].reshape(grid, H // grid, grid, W // grid, C).mean(axis=(1, 3)).ravel()


def nuisance_matrix(stimulus: Path, cells: list[dict[str, Any]], manifest: dict[str, dict[str, Any]], patch: int = 16, n_pix_pcs: int = 5) -> tuple[np.ndarray, list[str]]:
    n = max(c["row"] for c in cells) + 1
    pix, rest = np.zeros((n, 8 * 8 * 3)), np.zeros((n, 7 + 1 + 2 + 2 + 1))
    for c in cells:
        m = manifest.get(c["cell_id"], {})
        art = cell_artifact(stimulus, c, m)
        with np.load(art) as z:
            if "context_frames" in z:
                pix[c["row"]] = pixel_signature(np.asarray(z["context_frames"])[-1])
            pro = np.asarray(z["context_proprios"], dtype=np.float64)[-1] if "context_proprios" in z else np.zeros(7)
            act = float(np.linalg.norm(np.asarray(z["model_actions"], dtype=np.float64))) if "model_actions" in z else 0.0
        cen = np.asarray(m.get("target_centroid_xy") or [0.0, 0.0], dtype=np.float64)
        rest[c["row"]] = np.concatenate([pro[:7], [act], cen, np.floor(cen[::-1] / patch), [float(m.get("target_visible_pixels", 0.0))]])
    # PCA of the pixel signature (n << 192 dims) to a few components
    pc = pix - pix.mean(0)
    U, s, _ = np.linalg.svd(pc, full_matrices=False)
    pix_pcs = U[:, :n_pix_pcs] * s[:n_pix_pcs]
    N = np.column_stack([pix_pcs, rest])
    cols = [f"pix_pc{i}" for i in range(n_pix_pcs)] + [f"proprio_{i}" for i in range(7)] + ["action_l2", "egg_cx", "egg_cy", "egg_tok_row", "egg_tok_col", "egg_visible_px"]
    return N, cols


NUISANCE_LAM = 10.0
NUISANCE_CLIP = 3.0


def standardize_nuisance(N: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Standardize on the training rows, drop constant columns, clip at +-3 SD (a held-out scene can sit far
    outside the training range on a near-constant covariate and would otherwise dominate the ridge)."""
    mu, sd = N[train].mean(0), N[train].std(0)
    keep = sd > 1e-8
    Z = (N[:, keep] - mu[keep]) / sd[keep]
    return np.clip(Z, -NUISANCE_CLIP, NUISANCE_CLIP)


def residualize_nuisance(P: np.ndarray, N: np.ndarray, train: np.ndarray, lam: float = NUISANCE_LAM) -> np.ndarray:
    """P - ridge(N -> P) fitted on ``train`` rows only (scene-fold safe); N standardized/clipped on train."""
    Z = standardize_nuisance(N, train)
    Zt = Z[train]
    Pt = P[train]
    A = Zt.T @ Zt + lam * np.eye(Z.shape[1])
    B = np.linalg.solve(A, Zt.T @ (Pt - Pt.mean(0)))
    return P - Z @ B


# --------------------------------------------------------------------------- #
# Conceptor fitting on DiD rows
# --------------------------------------------------------------------------- #


def fit_conceptor_set(D: np.ndarray, A: np.ndarray, alpha: float, Dnull: np.ndarray | None = None, seed: int = 0) -> dict[str, Any]:
    """C_int, C_action, (C_null), C_safety = C_int AND NOT C_action (AND NOT C_null), controls; rows pre-scaled."""
    c_int = conceptor_factored(D, alpha, center=False)
    c_act = conceptor_factored(A, alpha, center=False)
    c_saf = conceptor_and_not(c_int, c_act)
    c_null = None
    if Dnull is not None and len(Dnull) >= 2:
        c_null = conceptor_factored(Dnull, alpha, center=False)
        c_saf = conceptor_and_not(c_saf, c_null)
    return {
        "int": c_int, "action": c_act, "null": c_null, "safety": c_saf,
        "random": matched_spectrum_random(c_saf, seed=seed),
        "rank_one": rank_one_projector(D),
    }


def readout_metric(kind: str, s: np.ndarray, y: np.ndarray) -> float:
    return auroc(s, y) if kind == "binary" else r_squared(s, y)


def loso_conceptor_readout(
    P: np.ndarray, N: np.ndarray, labels: dict[str, np.ndarray], meta: dict[str, np.ndarray], alpha_rule, lam: float, n_boot: int, seed: int,
    resid_nuisance: bool = True,
) -> dict[str, Any]:
    """Leave-one-scene-out: refit nuisance residualization + conceptors on the training scenes, project the held-out
    scene's cells with each operator, ridge readout trained on training projections.  ``alpha_rule(lam_int, d)``
    returns the aperture (fixed value or target-quota rule)."""
    n, d = P.shape
    pair = meta["pair_id"]
    scenes = np.unique(pair)
    ops = ("safety", "int", "rank_one", "random")
    oof = {t: {o: np.full(n, np.nan) for o in ops + ("nuisance", "nuisance+safety")} for t in TARGETS}
    alphas = []
    for held_scene in scenes:
        held = np.flatnonzero(pair == held_scene)
        train = np.flatnonzero(pair != held_scene)
        Pr = residualize_nuisance(P, N, train) if resid_nuisance else P
        Pc = scene_center(Pr, pair)
        mtr = {k: v[train] for k, v in meta.items()}
        D, A, _ = within_scene_did_rows(Pc[train], mtr)
        if len(D) < 2:
            continue
        scale = float(np.sqrt(np.mean(np.sum(D**2, axis=1)))) or 1.0
        D, A = D / scale, A / scale
        alpha = alpha_rule(conceptor_factored(D, 1.0, center=False)["lam"], d)
        alphas.append(alpha)
        cs = fit_conceptor_set(D, A, alpha, seed=seed)
        feats_by_target = {
            "contact": Pc, "force": Pc,
            "interaction": np.nan_to_num(quartet_residualize(Pr, meta, keep_hazard=True)),
        }
        Nz = standardize_nuisance(N, train)
        for t, kind in TARGETS.items():
            y = labels[t]
            ok_tr = train[np.isfinite(y[train])]
            if len(ok_tr) < 4 or len(np.unique(y[ok_tr])) < 2:
                continue
            F = feats_by_target[t]
            for o in ops:
                Z = conceptor_apply(F, cs[o])
                w, b = ridge_fit(Z[ok_tr], y[ok_tr], lam)
                oof[t][o][held] = ridge_predict(Z[held], w, b)
            w, b = ridge_fit(Nz[ok_tr], y[ok_tr], NUISANCE_LAM)
            oof[t]["nuisance"][held] = ridge_predict(Nz[held], w, b)
            Zs = np.column_stack([Nz, conceptor_apply(F, cs["safety"])])
            w, b = ridge_fit(Zs[ok_tr], y[ok_tr], NUISANCE_LAM)
            oof[t]["nuisance+safety"][held] = ridge_predict(Zs[held], w, b)
    out: dict[str, Any] = {"alpha_loso_mean": float(np.mean(alphas)) if alphas else float("nan")}
    for t, kind in TARGETS.items():
        y = labels[t]
        res: dict[str, Any] = {}
        ok = np.isfinite(y) & np.isfinite(oof[t]["safety"])
        if ok.sum() < 8 or len(np.unique(y[ok])) < 2:
            out[t] = {"status": "insufficient", "n_cells": int(ok.sum())}
            continue
        cl = pair[ok]
        for o in ops + ("nuisance", "nuisance+safety"):
            s = oof[t][o]
            res[o] = cluster_bootstrap(lambda idx, s=s: readout_metric(kind, s[ok][idx], y[ok][idx]), cl, n_boot=n_boot, seed=seed)
        for o, ref in (("rank_one", "safety_minus_rank_one"), ("random", "safety_minus_random"), ("int", "safety_minus_int")):
            res[ref] = cluster_bootstrap(
                lambda idx, o=o: readout_metric(kind, oof[t]["safety"][ok][idx], y[ok][idx]) - readout_metric(kind, oof[t][o][ok][idx], y[ok][idx]),
                cl, n_boot=n_boot, seed=seed,
            )
            res[ref]["ci_excludes_zero_positive"] = bool(np.isfinite(res[ref]["ci_low"]) and res[ref]["ci_low"] > 0)
        res["incremental_over_nuisance"] = cluster_bootstrap(
            lambda idx: readout_metric(kind, oof[t]["nuisance+safety"][ok][idx], y[ok][idx]) - readout_metric(kind, oof[t]["nuisance"][ok][idx], y[ok][idx]),
            cl, n_boot=n_boot, seed=seed,
        )
        res["incremental_over_nuisance"]["significant"] = bool(np.isfinite(res["incremental_over_nuisance"]["ci_low"]) and res["incremental_over_nuisance"]["ci_low"] > 0)
        thr = GATES["auroc"] if kind == "binary" else GATES["r2"]
        res["gate"] = {
            "metric_pass": bool(np.isfinite(res["safety"]["point"]) and res["safety"]["point"] >= thr),
            "beats_rank_one_ci": res["safety_minus_rank_one"]["ci_excludes_zero_positive"],
            "beats_random_ci": res["safety_minus_random"]["ci_excludes_zero_positive"],
            "incremental_pass": res["incremental_over_nuisance"]["significant"],
        }
        res["status"] = "ok"
        res["n_cells"] = int(ok.sum())
        out[t] = res
    return out


def stability_refits(D: np.ndarray, A: np.ndarray, alpha: float, seeds=(0, 1, 2), frac: float = 0.8) -> dict[str, Any]:
    n = len(D)
    fits = []
    for s in seeds:
        rng = np.random.default_rng(s)
        keep = np.sort(rng.choice(n, size=max(2, int(round(frac * n))), replace=False))
        fits.append(fit_conceptor_set(D[keep], A[keep], alpha, seed=s))
    sims = {k: [conceptor_similarity(fits[i][k], fits[j][k]) for i in range(len(fits)) for j in range(i + 1, len(fits))] for k in ("safety", "int")}
    return {"seeds": list(seeds), "safety_pairwise": sims["safety"], "safety_mean": float(np.mean(sims["safety"])), "int_mean": float(np.mean(sims["int"]))}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def write_conceptor_npz(path: Path, cs: dict[str, Any], *, mean: np.ndarray, scale: float, alpha: float, token_pool: str, site: str, step: int, group: str, quotas: dict[str, float], sweep: dict[str, Any]) -> Path:
    """Factored conceptor export for ``patch_site.py``.

    Arrays (all float32 unless noted), for each operator ``K`` in
    ``safety``, ``int``, ``action``, ``random`` (and ``null`` when fitted):
      ``K_eigvecs`` [r_K, d]  orthonormal rows,  ``K_mu`` [r_K] in [0, 1].
      The full operator is ``C_K = K_eigvecs.T @ diag(K_mu) @ K_eigvecs``  (d x d).
    ``rank_one_direction`` [d]: unit mean DiD direction (rank-one control; C = u u^T).
    File names: ``<site>__s<step>__<group>.npz`` (+ ``.json`` sidecar); site ids contain a dot, so no suffix logic.
    ``mean`` [d]: mean pooled activation of the discovery cells at this site/step/group
      (centre with it before steering and add it back: h' = mean + (h - mean) M).
    ``scale``: rows were divided by this RMS before the aperture was applied (alpha is
      defined on that scale; C itself is scale-free).
    ``alpha``, ``quota_*``, ``token_pool`` ("mean over <group> tokens" => apply the same
      operator to every token of the group at this imagined step), ``site``, ``step``, ``group``.
    Steering:  strengthen  h' = h @ ((1-b) I + b C_safety);  suppress  h' = h @ (I - b C_safety);
               reverse control: same with C_action; matched-spectrum control: C_random.
    A JSON sidecar carries the aperture sweep (quota / similarity per alpha).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = path.parent / (path.name + ".npz"), path.parent / (path.name + ".json")
    arrays: dict[str, Any] = {
        "mean": np.asarray(mean, dtype=np.float32), "scale": np.float32(scale), "alpha": np.float32(alpha),
        "token_pool": np.asarray(token_pool), "site": np.asarray(site), "step": np.int64(step), "group": np.asarray(group),
        "rank_one_direction": np.asarray(cs["rank_one"]["eigvecs"][0], dtype=np.float32),
    }
    for k in ("safety", "int", "action", "random", "null"):
        c = cs.get(k)
        if c is None:
            continue
        arrays[f"{k}_eigvecs"] = np.asarray(c["eigvecs"], dtype=np.float32)
        arrays[f"{k}_mu"] = np.asarray(c["mu"], dtype=np.float32)
    for k, v in quotas.items():
        arrays[f"quota_{k}"] = np.float32(v)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps({"site": site, "step": step, "group": group, "alpha": alpha, "quotas": quotas, "sweep": sweep,
                                     "apply": "h' = mean + (h - mean) @ ((1-b) I + b C_safety)   |   suppress: (I - b C_safety)"}, indent=1, default=_jd) + "\n")
    return npz_path


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Per site
# --------------------------------------------------------------------------- #


def analyze(site: str, acts, token_index, cells, token_sets, Q, labels, meta, N, args, out_dir: Path) -> list[dict[str, Any]]:
    results = []
    d = acts.shape[-1]
    for step in args.steps:
        for group in args.groups:
            P, tok_rows = pool_site(acts, token_index, cells, token_sets, step, group)
            entry: dict[str, Any] = {"site_id": site, "step": step, "group": group}
            valid = np.all(np.isfinite(P), axis=1)
            if valid.sum() < 8:
                entry["status"] = "no_tokens"
                results.append(entry)
                continue
            P = np.where(valid[:, None], P, np.nan_to_num(P[valid].mean(0)))
            train = np.arange(len(P))
            Pr = residualize_nuisance(P, N, train)
            Pc = scene_center(Pr, meta["pair_id"])
            D, A, ids = within_scene_did_rows(Pc, meta)
            if len(D) < 3:
                entry["status"] = "too_few_quartets"
                results.append(entry)
                continue
            scale = float(np.sqrt(np.mean(np.sum(D**2, axis=1)))) or 1.0
            Ds, As = D / scale, A / scale
            # per-token DiD rows (matched token ids within each quartet)
            tok_D = per_token_did_rows(acts, token_index, cells, token_sets, Q, step, group)
            # aperture sweep on pooled rows
            base = conceptor_factored(Ds, 1.0, center=False)
            sweep = {}
            for a in ALPHAS:
                cs = fit_conceptor_set(Ds, As, a, seed=args.seed)
                sweep[str(a)] = {
                    "quota_int": conceptor_quota(cs["int"], d), "quota_action": conceptor_quota(cs["action"], d), "quota_safety": conceptor_quota(cs["safety"], d),
                    "overlap_int_action": conceptor_similarity(cs["int"], cs["action"]),
                    "safety_vs_int": conceptor_similarity(cs["safety"], cs["int"]),
                    "random_vs_safety": conceptor_similarity(cs["random"], cs["safety"]),
                    "rank_one_vs_safety": conceptor_similarity(cs["rank_one"], cs["safety"]),
                }
            alpha_sel = alpha_for_quota(base["lam"], d, args.target_dims / d)
            cs = fit_conceptor_set(Ds, As, alpha_sel, seed=args.seed)
            quotas = {"int": conceptor_quota(cs["int"], d), "action": conceptor_quota(cs["action"], d), "safety": conceptor_quota(cs["safety"], d)}
            entry.update({
                "status": "ok", "n_scenes": int(len(D)), "alpha_selected": alpha_sel, "target_dims": args.target_dims,
                "quotas": quotas, "overlap_int_action": conceptor_similarity(cs["int"], cs["action"]),
                "safety_rank_mu_gt_half": int(np.sum(cs["safety"]["mu"] > 0.5)),
                "safety_mu_top": [float(v) for v in np.sort(cs["safety"]["mu"])[::-1][:8]],
                "int_lambda_top": [float(v) for v in base["lam"][:8]],
                "sweep": sweep,
                "stability": stability_refits(Ds, As, alpha_sel, seeds=tuple(args.stability_seeds)),
                "per_token": {},
            })
            if tok_D is not None and len(tok_D) >= 8:
                ts = float(np.sqrt(np.mean(np.sum(tok_D**2, axis=1)))) or 1.0
                ct = conceptor_factored(tok_D / ts, alpha_sel, center=False)
                entry["per_token"] = {"n_rows": int(len(tok_D)), "quota_int": conceptor_quota(ct, d), "similarity_to_pooled_int": conceptor_similarity(ct, cs["int"]),
                                      "quota_int_by_alpha": {str(a): conceptor_quota(conceptor_factored(tok_D / ts, a, center=False), d) for a in ALPHAS}}
            entry["readout"] = loso_conceptor_readout(P, N, labels, meta, lambda lam_, d_: alpha_for_quota(lam_, d_, args.target_dims / d_), args.lam, args.n_boot, args.seed)
            entry["readout_alpha_sweep"] = {str(a): {t: {k: v.get("point") for k, v in r.items() if isinstance(v, dict) and "point" in v} for t, r in loso_conceptor_readout(P, N, labels, meta, lambda lam_, d_, a=a: a, args.lam, max(50, args.n_boot // 5), args.seed).items() if isinstance(r, dict) and r.get("status") == "ok"} for a in (args.sweep_readout_alphas or [])}
            entry["conceptor_file"] = str(write_conceptor_npz(out_dir / "conceptors" / f"{site}__s{step}__{group}", cs, mean=P.mean(0), scale=scale, alpha=alpha_sel, token_pool=f"mean over {group} tokens", site=site, step=step, group=group, quotas=quotas, sweep=sweep))
            entry["_c_int"] = cs["int"]  # kept in memory for cross-step / cross-layer similarity, stripped before writing
            entry["_c_safety"] = cs["safety"]
            results.append(entry)
    return results


def per_token_did_rows(acts, token_index, cells, token_sets, Q, step, group) -> np.ndarray | None:
    rows = []
    for pid, q in Q.items():
        if not all(k in q for k in CELL_ORDER):
            continue
        want = token_sets[(pid, step, group)]
        per = {}
        for k in CELL_ORDER:
            r = q[k]
            ti = token_index[r, step]
            m = np.isin(ti, want) & (ti >= 0)
            per[k] = dict(zip(ti[m].tolist(), acts[r, step, m].astype(np.float64)))
        common = set(per[(0, 0)]) & set(per[(0, 1)]) & set(per[(1, 0)]) & set(per[(1, 1)])
        for tok in sorted(common):
            rows.append(per[(1, 1)][tok] - per[(1, 0)][tok] - per[(0, 1)][tok] + per[(0, 0)][tok])
    return np.asarray(rows) if rows else None


def cross_similarity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """C_int similarity across imagined steps (same site/group) and across adjacent layers (same hook/step/group)."""
    by = {(e["site_id"], e["step"], e["group"]): e for e in entries if e.get("status") == "ok"}
    steps, layers = {}, {}
    for (site, step, group), e in by.items():
        for (site2, step2, group2), e2 in by.items():
            if site2 == site and group2 == group and step2 == step + 1:
                steps[f"{site}|{group}|s{step}->s{step2}"] = {"int": conceptor_similarity(e["_c_int"], e2["_c_int"]), "safety": conceptor_similarity(e["_c_safety"], e2["_c_safety"])}
        hook = site.split(".", 1)[1] if "." in site else ""
        try:
            L = int(site.split(".")[0][1:])
        except ValueError:
            continue
        nxt = f"L{L + 1:02d}.{hook}"
        e2 = by.get((nxt, step, group))
        if e2 is not None:
            layers[f"{site}->{nxt}|s{step}|{group}"] = {"int": conceptor_similarity(e["_c_int"], e2["_c_int"]), "safety": conceptor_similarity(e["_c_safety"], e2["_c_safety"])}
    return {"across_steps": steps, "across_adjacent_layers": layers}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--groups", nargs="*", default=["gripper_corridor", "egg"])
    ap.add_argument("--target-dims", type=float, default=4.0, help="aperture selection: quota * d (effective dims of C_int)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability-seeds", type=int, nargs=3, default=[0, 1, 2])
    ap.add_argument("--sweep-readout-alphas", type=float, nargs="*", default=[3.0, 30.0])
    args = ap.parse_args(argv)

    t0 = time.time()
    idx = json.loads((args.dump / "activations" / "index.json").read_text())
    cells = idx["cells"] if isinstance(idx, dict) else idx
    if args.discovery_seeds is not None:
        allowed = {int(v) for v in args.discovery_seeds.read_text().split() if v.strip()}
        cells = [c for c in cells if int(c["seed"]) in allowed]
    # compact rows for the selected cells; keep the dump row for slicing the activation arrays
    cells = sorted(cells, key=lambda c: int(c["row"]))
    for i, c in enumerate(cells):
        c["dump_row"] = int(c["row"])
        c["row"] = i
    dump_rows = np.asarray([c["dump_row"] for c in cells])
    n_steps = int(idx.get("n_steps", 3)) if isinstance(idx, dict) else 3
    offset = int(idx.get("group_frame_offset", 0)) if isinstance(idx, dict) else 0
    manifest = {}
    mp = args.stimulus / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                manifest[r["cell_id"]] = r
    for c in cells:
        c.setdefault("target_bbox_xyxy", manifest.get(c["cell_id"], {}).get("target_bbox_xyxy"))
    Q = quartets(cells)
    labels = interaction_labels(cells, manifest, Q)
    n = max(c["row"] for c in cells) + 1
    contact = np.full(n, np.nan)
    for c in cells:
        v = manifest.get(c["cell_id"], {}).get("egg_robot_contact_count")
        contact[c["row"]] = float(v > 0) if v is not None else np.nan
    labels["contact"] = contact
    meta = {"pair_id": np.asarray([""] * n, dtype=object), "hazard": np.zeros(n, dtype=int), "action": np.zeros(n, dtype=int)}
    for c in cells:
        meta["pair_id"][c["row"]] = c["pair_id"]
        meta["hazard"][c["row"]] = int(c["hazard"])
        meta["action"][c["row"]] = int(c["candidate_action"])
    meta["pair_id"] = meta["pair_id"].astype(str)
    N, ncols = nuisance_matrix(args.stimulus, cells, manifest)
    token_sets = group_token_sets(args.stimulus, cells, tuple(args.groups), n_steps, offset)
    sites = args.sites or sorted(p.stem for p in (args.dump / "activations").glob("*.npz"))
    args.out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for i, site in enumerate(sites):
        with np.load(args.dump / "activations" / f"{site}.npz") as z:
            acts = np.asarray(z["acts"])[dump_rows]
            tix = (np.asarray(z["token_index"]) if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), (len(dump_rows),) + acts.shape[1:3]))[dump_rows] if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3])
        entries.extend(analyze(site, acts, tix, cells, token_sets, Q, labels, meta, N, args, args.out))
        print(f"[{i + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    xs = cross_similarity(entries)
    for e in entries:
        e.pop("_c_int", None)
        e.pop("_c_safety", None)
    ok = [e for e in entries if e.get("status") == "ok"]
    quota_table = {f"s{s}|{g}": {e["site_id"]: {"quota_int": e["quotas"]["int"], "quota_safety": e["quotas"]["safety"], "alpha": e["alpha_selected"], "overlap_int_action": e["overlap_int_action"]}
                                 for e in ok if e["step"] == s and e["group"] == g} for s in args.steps for g in args.groups}
    ranked = {}
    for t in TARGETS:
        rows = []
        for e in ok:
            r = e["readout"].get(t, {})
            if r.get("status") != "ok":
                continue
            rows.append({"site_id": e["site_id"], "step": e["step"], "group": e["group"], "safety": r["safety"]["point"], "rank_one": r["rank_one"]["point"], "random": r["random"]["point"],
                         "int": r["int"]["point"], "nuisance": r["nuisance"]["point"], "safety_minus_rank_one": r["safety_minus_rank_one"]["point"], "beats_rank_one_ci": r["gate"]["beats_rank_one_ci"],
                         "beats_random_ci": r["gate"]["beats_random_ci"], "incremental": r["incremental_over_nuisance"]["point"], "incremental_pass": r["gate"]["incremental_pass"], "metric_pass": r["gate"]["metric_pass"],
                         "quota_safety": e["quotas"]["safety"], "overlap_int_action": e["overlap_int_action"]})
        rows.sort(key=lambda r: -(r["safety"] if np.isfinite(r["safety"]) else -np.inf))
        ranked[t] = rows
    report = {
        "protocol": "cgs-geometry-conceptor-v0.1", "dump": str(args.dump), "stimulus": str(args.stimulus), "n_cells": int(n), "n_scenes": len(Q),
        "steps": args.steps, "groups": args.groups, "alphas": list(ALPHAS), "target_dims": args.target_dims, "lam": args.lam, "nuisance_columns": ncols,
        "row_scaling": "DiD rows divided by their RMS norm before the aperture; alpha is on that scale",
        "quota_tables": quota_table, "ranked_by_target": ranked, "cross_similarity": xs, "entries": entries, "runtime_s": time.time() - t0,
        "interpretation_scope": "Descriptive conceptor geometry (COAST entrant); causal steering/suppression happens in patch_site.py.",
    }
    (args.out / "conceptor_map.json").write_text(json.dumps(report, indent=1, default=_jd) + "\n")
    for t, rows in ranked.items():
        print(f"== {t}: top by held-out C_safety readout ==")
        for r in rows[:8]:
            print(f"  {r['site_id']:>14s} s{r['step']} {r['group']:>16s} safety={r['safety']:.3f} rank1={r['rank_one']:.3f} rand={r['random']:.3f} nuis={r['nuisance']:.3f} "
                  f"d_r1={r['safety_minus_rank_one']:+.3f}{'*' if r['beats_rank_one_ci'] else ''} incr={r['incremental']:+.3f}{'*' if r['incremental_pass'] else ''} q={r['quota_safety']:.4f} ov={r['overlap_int_action']:.2f}")
    return report


if __name__ == "__main__":
    main()
