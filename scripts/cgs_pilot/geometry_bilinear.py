#!/usr/bin/env python3
"""Bilinear relational probe arm (design doc JEPA rep geometry.md 2.4 / hypothesis E4).

A linear readout on concatenated tokens cannot express "egg content x route
content".  Per site x imagined step, with route-token mean ``z_g`` (gripper_corridor
group) and egg-token mean ``z_e`` (egg group), nuisance-residualized, fit
``y = z_g^T W z_e + b`` with ``W = U V^T`` of rank r in {1, 2} by alternating ridge
(lambda picked by inner scene-fold CV on the train fold), leave-one-scene-out.

Targets: interaction contrast, contact and force on scene-centred features (the
relational term a*h arises as A_g^T W H_e), the H1-vs-H0' null contrast when
hazard-2 cells exist, and the quartet-residualized interaction as a documented
negative control (bilinear forms are even in the contrast there and the
cross-covariance cancels exactly).  Baselines at comparable parameter budget: linear ridge on
[z_g, z_e] (2d params = rank-1 bilinear), linear ridge on the elementwise product
z_g * z_e (d), linear on [z_g, z_e, z_g * z_e] (3d), RBF kernel ridge on the
concatenation; plus the quartet-shuffled null (labels permuted among the four
cells of each scene) for the bilinear readout.  Report delta = bilinear - best
baseline with a scene-clustered bootstrap CI and flag CI > 0.

Note: a per-token-pair bilinear score summed over the two groups equals the
bilinear score of the group means (bilinearity), so the "per-token-pair" variant
reported here is the max over token pairs of z_t^T W z_s with W fit on the means.
References: arXiv:2606.09646 (probe capacity on physics readouts), design doc 2.4.
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
from geometry_conceptor import nuisance_matrix, residualize_nuisance  # noqa: E402
from geometry_localize import KernelRidgeRBF, collect_linear_entries, group_token_sets, interaction_labels, pool_site, quartets  # noqa: E402
from geometry_models import (  # noqa: E402
    auroc,
    bilinear_fit,
    bilinear_predict,
    cluster_bootstrap,
    quartet_residualize,
    r_squared,
    ridge_fit,
    ridge_predict,
    scene_center,
    scene_folds,
)

CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
TARGETS = {"interaction": "binary", "contact": "binary", "force": "continuous", "null_contrast": "binary", "interaction_residualized": "binary"}
LAMS = (0.01, 0.1, 1.0, 10.0)
BASELINES = ("linear_concat", "linear_product", "linear_concat_product", "rbf_concat")


def metric(kind: str, s: np.ndarray, y: np.ndarray) -> float:
    return auroc(s, y) if kind == "binary" else r_squared(s, y)


def inner_cv_lambda(fit_fn, score_fn, Zg, Ze, y, pair, kind, lams=LAMS, n_folds=4, seed=0) -> float:
    best, best_l = -np.inf, lams[0]
    for lam in lams:
        oof = np.full(len(y), np.nan)
        for held in scene_folds(pair, n_folds, seed):
            tr = np.setdiff1d(np.arange(len(y)), held)
            if len(np.unique(y[tr])) < 2:
                continue
            model = fit_fn(Zg[tr], Ze[tr], y[tr], lam)
            oof[held] = score_fn(model, Zg[held], Ze[held])
        ok = np.isfinite(oof)
        # a collapsed (constant) readout is not a valid choice; ties go to the smaller lambda
        if ok.sum() <= 4 or len(np.unique(y[ok])) < 2 or np.std(oof[ok]) < 1e-10:
            continue
        m = metric(kind, oof[ok], y[ok])
        if np.isfinite(m) and m > best:
            best, best_l = m, lam
    return best_l


def make_fitters(rank: int):
    fitters = {
        f"bilinear_r{rank}": (
            lambda Zg, Ze, y, lam: bilinear_fit(Zg, Ze, y, rank, lam),
            lambda m, Zg, Ze: bilinear_predict(Zg, Ze, *m),
        ),
        "linear_concat": (
            lambda Zg, Ze, y, lam: ridge_fit(np.concatenate([Zg, Ze], 1), y, lam),
            lambda m, Zg, Ze: ridge_predict(np.concatenate([Zg, Ze], 1), *m),
        ),
        "linear_product": (
            lambda Zg, Ze, y, lam: ridge_fit(Zg * Ze, y, lam),
            lambda m, Zg, Ze: ridge_predict(Zg * Ze, *m),
        ),
        "linear_concat_product": (
            lambda Zg, Ze, y, lam: ridge_fit(np.concatenate([Zg, Ze, Zg * Ze], 1), y, lam),
            lambda m, Zg, Ze: ridge_predict(np.concatenate([Zg, Ze, Zg * Ze], 1), *m),
        ),
        "rbf_concat": (
            lambda Zg, Ze, y, lam: KernelRidgeRBF(lam=lam).fit(np.concatenate([Zg, Ze], 1), y, {}),
            lambda m, Zg, Ze: m.score(np.concatenate([Zg, Ze], 1)),
        ),
    }
    return fitters


def pca_project(Ztr: np.ndarray, Zte: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Label-free PCA of the training rows to k dims (both halves of the probe get their own basis).  With ~44
    training cells in 1024-d a rank-2 bilinear form (4d parameters) is hopelessly under-determined; the probe is
    therefore fit on the top-k principal coordinates of each group, and every baseline sees the same features."""
    if k is None or k >= Ztr.shape[1]:
        return Ztr, Zte
    mu = Ztr.mean(0)
    _, _, vt = np.linalg.svd(Ztr - mu, full_matrices=False)
    B = vt[:k].T
    A, Bt = (Ztr - mu) @ B, (Zte - mu) @ B
    # unit variance per coordinate (train fold) so bilinear products are O(1) and the ridge grid is meaningful
    sd = A.std(0) + 1e-12
    return A / sd, Bt / sd


def loso_readouts(Zg, Ze, y, pair, kind, ranks=(1, 2), seed=0, pca_dims: int | None = None) -> dict[str, np.ndarray]:
    """Out-of-fold predictions per model (lambda chosen by inner CV on each train fold)."""
    ok_all = np.isfinite(y)
    scenes = np.unique(pair[ok_all])
    models = {}
    for r in ranks:
        models.update({k: v for k, v in make_fitters(r).items() if k.startswith("bilinear")})
    models.update({k: v for k, v in make_fitters(1).items() if not k.startswith("bilinear")})
    oof = {name: np.full(len(y), np.nan) for name in models}
    for s in scenes:
        te = (pair == s) & ok_all
        tr = (pair != s) & ok_all
        if tr.sum() < 8 or len(np.unique(y[tr])) < 2:
            continue
        Zg_tr, Zg_te = pca_project(Zg[tr], Zg[te], pca_dims)
        Ze_tr, Ze_te = pca_project(Ze[tr], Ze[te], pca_dims)
        for name, (fit_fn, score_fn) in models.items():
            lam = inner_cv_lambda(fit_fn, score_fn, Zg_tr, Ze_tr, y[tr], pair[tr], kind, seed=seed)
            m = fit_fn(Zg_tr, Ze_tr, y[tr], lam)
            oof[name][te] = score_fn(m, Zg_te, Ze_te)
    return oof


def quartet_shuffled_null(Zg, Ze, y, pair, hz, ac, kind, rank, n_shuffles, seed, pca_dims: int | None = None) -> np.ndarray:
    """Permute the labels among the four cells of each scene, refit the bilinear LOSO readout, return null AUROCs.
    Lambda is fixed to the value chosen on the unshuffled data (inner CV on all scenes) to keep the null cheap."""
    rng = np.random.default_rng(seed)
    fit_fn, score_fn = make_fitters(rank)[f"bilinear_r{rank}"]
    ok = np.isfinite(y)
    Zg, Ze = pca_project(Zg[ok], Zg, pca_dims)[1], pca_project(Ze[ok], Ze, pca_dims)[1]
    lam = inner_cv_lambda(fit_fn, score_fn, Zg[ok], Ze[ok], y[ok], pair[ok], kind, seed=seed)
    scenes = np.unique(pair[ok])
    null = []
    for _ in range(n_shuffles):
        yp = y.copy()
        for s in scenes:
            idx = np.flatnonzero((pair == s) & ok)
            yp[idx] = y[idx][rng.permutation(len(idx))]
        oof = np.full(len(y), np.nan)
        for s in scenes:
            te = (pair == s) & ok
            tr = (pair != s) & ok
            if len(np.unique(yp[tr])) < 2:
                continue
            m = fit_fn(Zg[tr], Ze[tr], yp[tr], lam)
            oof[te] = score_fn(m, Zg[te], Ze[te])
        f = np.isfinite(oof) & ok
        null.append(metric(kind, oof[f], yp[f]) if len(np.unique(yp[f])) > 1 else np.nan)
    return np.asarray(null)


def per_token_pair_max(acts, token_index, cells, token_sets, Q, step, U, V, b, groups=("gripper_corridor", "egg")) -> np.ndarray:
    """Max over (route token, egg token) pairs of z_t^T W z_s + b per cell, W = U V^T fit on the group means."""
    out = np.full(len(cells), np.nan)
    for c in cells:
        r = c["row"]
        ti = token_index[r, step]
        g = np.isin(ti, token_sets[(c["pair_id"], step, groups[0])]) & (ti >= 0)
        e = np.isin(ti, token_sets[(c["pair_id"], step, groups[1])]) & (ti >= 0)
        if g.any() and e.any():
            A = acts[r, step, g].astype(np.float64) @ U
            B = acts[r, step, e].astype(np.float64) @ V
            out[r] = float((A @ B.T).max()) + b
    return out


def analyze_site(site, acts, token_index, cells, token_sets, Q, labels, meta, N, args) -> list[dict[str, Any]]:
    results = []
    n = len(cells)
    pair, hz, ac = meta["pair_id"], meta["hazard"], meta["action"]
    train_all = np.arange(n)
    for step in args.steps:
        entry: dict[str, Any] = {"site_id": site, "step": step, "groups": [args.route_group, args.egg_group]}
        Pg, _ = pool_site(acts, token_index, cells, token_sets, step, args.route_group)
        Pe, _ = pool_site(acts, token_index, cells, token_sets, step, args.egg_group)
        valid = np.all(np.isfinite(Pg), 1) & np.all(np.isfinite(Pe), 1)
        if valid.sum() < 8:
            entry["status"] = "no_tokens"
            results.append(entry)
            continue
        Pg = np.where(valid[:, None], Pg, np.nan_to_num(Pg[valid].mean(0)))
        Pe = np.where(valid[:, None], Pe, np.nan_to_num(Pe[valid].mean(0)))
        Rg, Re = residualize_nuisance(Pg, N, train_all), residualize_nuisance(Pe, N, train_all)
        # scale each group's rows so products are O(1)
        sg, se = np.sqrt(np.mean(np.sum(Rg**2, 1))) + 1e-12, np.sqrt(np.mean(np.sum(Re**2, 1))) + 1e-12
        # Interaction target on SCENE-CENTRED features (main effects retained): the relational term is
        # A_g^T W H_e ~ a*h, i.e. route action content x egg hazard content.  On quartet-residualized features
        # ((h-1/2)H +/- I/4) every bilinear form is even in the contrast and the cross-covariance cancels exactly,
        # so that variant ("interaction_residualized") is kept only as the documented negative control.
        sc = (scene_center(Rg, pair) / sg, scene_center(Re, pair) / se)
        feats = {
            "interaction": sc, "contact": sc, "force": sc, "null_contrast": sc,
            "interaction_residualized": (np.nan_to_num(quartet_residualize(Rg, meta)) / sg, np.nan_to_num(quartet_residualize(Re, meta)) / se),
        }
        entry["targets"] = {}
        for tname, kind in TARGETS.items():
            y = labels.get("interaction" if tname == "interaction_residualized" else tname)
            if y is None or np.isfinite(y).sum() < 8 or len(np.unique(y[np.isfinite(y)])) < 2:
                entry["targets"][tname] = {"status": "unavailable" if tname == "null_contrast" else "insufficient"}
                continue
            Zg, Ze = feats[tname]
            oof = loso_readouts(Zg, Ze, y, pair, kind, ranks=tuple(args.ranks), seed=args.seed, pca_dims=args.pca_dims)
            ok = np.isfinite(y) & np.all(np.stack([np.isfinite(v) for v in oof.values()]), 0)
            if ok.sum() < 8 or len(np.unique(y[ok])) < 2:
                entry["targets"][tname] = {"status": "insufficient"}
                continue
            cl = pair[ok]
            res: dict[str, Any] = {"status": "ok", "n_cells": int(ok.sum())}
            for name, s in oof.items():
                res[name] = cluster_bootstrap(lambda idx, s=s: metric(kind, s[ok][idx], y[ok][idx]), cl, n_boot=args.n_boot, seed=args.seed)
            base_names = [b for b in BASELINES if b in oof]
            best_base = max(base_names, key=lambda b: res[b]["point"] if np.isfinite(res[b]["point"]) else -np.inf)
            res["best_baseline"] = best_base
            for r in args.ranks:
                name = f"bilinear_r{r}"
                delta = cluster_bootstrap(
                    lambda idx, s=oof[name], t=oof[best_base]: metric(kind, s[ok][idx], y[ok][idx]) - metric(kind, t[ok][idx], y[ok][idx]),
                    cl, n_boot=args.n_boot, seed=args.seed,
                )
                delta["ci_excludes_zero_positive"] = bool(np.isfinite(delta["ci_low"]) and delta["ci_low"] > 0)
                res[f"delta_{name}_minus_best_baseline"] = delta
                # quartet-shuffled null
                null = quartet_shuffled_null(Zg, Ze, y, pair, hz, ac, kind, r, args.n_shuffles, args.seed, pca_dims=args.pca_dims)
                obs = res[name]["point"]
                res[f"{name}_shuffled_null"] = {"p": float((np.sum(null >= obs) + 1) / (np.sum(np.isfinite(null)) + 1)), "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null))}
                # per-token-pair max variant (W fit on all scenes' means; descriptive, in-sample W)
                if tname == "interaction":
                    # per-token-pair max needs W in the native space: fit on PCA coords and map back through the bases
                    mug, mue = Zg[ok].mean(0), Ze[ok].mean(0)
                    Bg = np.linalg.svd(Zg[ok] - mug, full_matrices=False)[2][: args.pca_dims].T if args.pca_dims else np.eye(Zg.shape[1])
                    Be = np.linalg.svd(Ze[ok] - mue, full_matrices=False)[2][: args.pca_dims].T if args.pca_dims else np.eye(Ze.shape[1])
                    Zg_p, Ze_p = (Zg[ok] - mug) @ Bg, (Ze[ok] - mue) @ Be
                    lam = inner_cv_lambda(*make_fitters(r)[name], Zg_p, Ze_p, y[ok], pair[ok], kind, seed=args.seed)
                    Up, Vp, b = bilinear_fit(Zg_p, Ze_p, y[ok], r, lam)
                    U, V = Bg @ Up, Be @ Vp
                    # apply to per-token rows: residualize per token with the same nuisance model is not available here;
                    # use raw token rows scaled by the pooled scales (descriptive only)
                    pm = per_token_pair_max(acts, token_index, cells, token_sets, Q, step, U / sg, V / se, b, (args.route_group, args.egg_group))
                    f2 = ok & np.isfinite(pm)
                    res[f"{name}_per_token_pair_max_insample"] = metric(kind, pm[f2], y[f2]) if f2.sum() > 4 else float("nan")
            entry["targets"][tname] = res
        entry["status"] = "ok"
        results.append(entry)
    return results


def rank_table(entries, linear, steps=(0, 1), target="interaction", rank=2) -> list[dict[str, Any]]:
    rows = []
    for e in entries:
        if e.get("status") != "ok" or e["step"] not in steps:
            continue
        r = e["targets"].get(target, {})
        if r.get("status") != "ok":
            continue
        name = f"bilinear_r{rank}"
        d = r[f"delta_{name}_minus_best_baseline"]
        lin = linear.get((e["site_id"], e["step"], "gripper_corridor"), {})
        rows.append({
            "site_id": e["site_id"], "step": e["step"], "bilinear": r[name]["point"], "bilinear_ci": [r[name]["ci_low"], r[name]["ci_high"]],
            "bilinear_r1": r["bilinear_r1"]["point"] if "bilinear_r1" in r else None,
            "best_baseline": r["best_baseline"], "best_baseline_value": r[r["best_baseline"]]["point"],
            "linear_concat": r["linear_concat"]["point"], "linear_product": r["linear_product"]["point"], "rbf_concat": r["rbf_concat"]["point"],
            "delta": d["point"], "delta_ci": [d["ci_low"], d["ci_high"]], "ci_excludes_zero": d["ci_excludes_zero_positive"],
            "shuffled_null_p": r[f"{name}_shuffled_null"]["p"], "shuffled_null_mean": r[f"{name}_shuffled_null"]["null_mean"],
            "per_token_pair_max_insample": r.get(f"{name}_per_token_pair_max_insample"),
            "linear_map_t": lin.get("t"), "linear_map_p_maxt": lin.get("p_maxt_fwer"),
        })
    rows.sort(key=lambda x: -(x["delta"] if np.isfinite(x["delta"]) else -np.inf))
    return rows


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--linear-map", type=Path, default=None)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--route-group", default="gripper_corridor")
    ap.add_argument("--egg-group", default="egg")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--pca-dims", type=int, default=8, help="label-free PCA per group on the train fold before the probe (0 = none)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-shuffles", type=int, default=99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t0 = time.time()
    args.pca_dims = args.pca_dims or None
    idx = json.loads((args.dump / "activations" / "index.json").read_text())
    cells = idx["cells"] if isinstance(idx, dict) else idx
    if args.discovery_seeds is not None:
        allowed = {int(v) for v in args.discovery_seeds.read_text().split() if v.strip()}
        cells = [c for c in cells if int(c["seed"]) in allowed]
    cells = sorted(cells, key=lambda c: int(c["row"]))
    dump_rows = np.asarray([int(c["row"]) for c in cells])
    for i, c in enumerate(cells):
        c["row"] = i
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
    n = len(cells)
    contact = np.full(n, np.nan)
    for c in cells:
        v = manifest.get(c["cell_id"], {}).get("egg_robot_contact_count")
        contact[c["row"]] = float(v > 0) if v is not None else np.nan
    labels["contact"] = contact
    # null contrast: h==1 vs h==2 cells (nan elsewhere) when hazard-2 cells exist
    hz_all = np.asarray([int(c["hazard"]) for c in cells])
    if np.any(hz_all == 2):
        yn = np.full(n, np.nan)
        yn[hz_all == 1] = 1.0
        yn[hz_all == 2] = 0.0
        labels["null_contrast"] = yn
    meta = {"pair_id": np.asarray([c["pair_id"] for c in cells]).astype(str), "hazard": hz_all, "action": np.asarray([int(c["candidate_action"]) for c in cells])}
    N, ncols = nuisance_matrix(args.stimulus, cells, manifest)
    token_sets = group_token_sets(args.stimulus, cells, (args.route_group, args.egg_group), n_steps, offset)
    linear: dict = {}
    if args.linear_map and args.linear_map.exists():
        collect_linear_entries(json.loads(args.linear_map.read_text()), linear)
    sites = args.sites or sorted(p.stem for p in (args.dump / "activations").glob("*.npz"))
    entries: list[dict[str, Any]] = []
    for i, site in enumerate(sites):
        with np.load(args.dump / "activations" / f"{site}.npz") as z:
            acts = np.asarray(z["acts"])[dump_rows]
            tix = np.asarray(z["token_index"])[dump_rows] if "token_index" in z else np.broadcast_to(np.arange(acts.shape[2]), acts.shape[:3])
        entries.extend(analyze_site(site, acts, tix, cells, token_sets, Q, labels, meta, N, args))
        print(f"[{i + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    tables = {f"{t}|r{r}": rank_table(entries, linear, steps=tuple(s for s in (0, 1) if s in args.steps), target=t, rank=r) for t in TARGETS for r in args.ranks}
    report = {
        "protocol": "cgs-geometry-bilinear-v0.1", "dump": str(args.dump), "stimulus": str(args.stimulus), "n_cells": n, "n_scenes": len(Q),
        "scenes_with_hazard2": int(np.sum(hz_all == 2) // 2), "steps": args.steps, "route_group": args.route_group, "egg_group": args.egg_group,
        "ranks": args.ranks, "pca_dims": args.pca_dims, "lambda_grid": list(LAMS), "baselines": list(BASELINES), "nuisance_columns": ncols,
        "note": "per-token-pair bilinear summed over pairs equals the bilinear of the means; the per-token variant reported is the max over pairs (descriptive, in-sample W)",
        "ranked_tables": tables, "entries": entries, "runtime_s": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bilinear_map.json").write_text(json.dumps(report, indent=1, default=_jd) + "\n")
    for key, rows in tables.items():
        if not rows:
            continue
        print(f"== {key} ==")
        for r in rows[:8]:
            print(f"  {r['site_id']:>14s} s{r['step']} bilinear={r['bilinear']:.3f} best_base={r['best_baseline']}={r['best_baseline_value']:.3f} delta={r['delta']:+.3f} [{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}]{'*' if r['ci_excludes_zero'] else ''} shuffle_p={r['shuffled_null_p']:.3f} map_t={r['linear_map_t']}")
    return report


if __name__ == "__main__":
    main()
