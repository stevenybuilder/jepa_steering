#!/usr/bin/env python3
"""Phase-2 geometry-sonar tournament at frozen predictor sites (HANDOFF step 12).

Reads the activation dump written by ``localize_interaction.py``::

    <dump>/activations/index.json          # [{pair_id, hazard, candidate_action, cell_id, seed}, ...]
    <dump>/activations/<site_id>.npz       # acts float16 [n_cells, n_imagined_steps, n_tokens, d]

and the stimulus directory (``manifest.jsonl`` + ``summary.json`` + ``cells/*.npz``)
for physical labels and nuisance covariates, then compares coordinate models
with identical scene-clustered splits and equal representation budgets.

Per site and model it reports (design "Geometry gate"):

- held-out contact AUROC (gate >= 0.75) and clearance/force R^2 (gate >= 0.30);
- incremental prediction over nuisance covariates (proprio, action magnitude,
  egg pixel centroid/visibility) with scene-clustered bootstrap CIs;
- within-scene interaction-sign agreement (does the readout reproduce the
  hazard x action DiD sign inside each scene);
- chart stability across three discovery resamples;
- cross-family transport when >1 ``risk_family`` is present, else it is
  labelled ``task_local`` (single family; transport untested).

Discovery seeds fit and select; confirmation seeds are scored once. Nothing is
fit on calibration seeds 101/102 unless they are explicitly listed.

Outputs ``<out>/tournament.json`` and per-model edit artifacts under
``<out>/edits/`` (format documented in ``geometry_models.write_edit_artifact``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_models import (  # noqa: E402
    MODELS,
    JacobianLocal,
    LowRankSubspace,
    auroc,
    clean_distribution,
    cluster_bootstrap,
    edit_cosine,
    effective_rank,
    on_manifold_scores,
    quartet_residualize,
    r_squared,
    random_equal_norm_edits,
    ridge_fit,
    ridge_predict,
    scene_center,
    scene_folds,
    write_edit_artifact,
)
from geometry_frames import collect_frames, frame_table  # noqa: E402

CALIBRATION_SEEDS = {101, 102}
GATES = {"auroc": 0.75, "r2": 0.30, "stability": 0.70}
TARGETS = {
    # name: (manifest field, kind)
    "contact": ("egg_robot_contact_count", "binary"),
    "force": ("max_normal_force_n", "continuous"),
    "clearance": ("gripper_to_target_approach_m", "continuous"),
    # within-scene interaction: sign(scene DiD of contact) * (2h-1)(2a-1) -> {0,1}; nan where the scene has no DiD
    "interaction": (None, "binary"),  # evaluated on quartet-residualized features (see quartet_residualize)
}
RANK_SWEEP = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
PATCH_PX = 16


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_index(dump: Path) -> list[dict[str, Any]]:
    """index.json is either a bare list of cells or (localization dump) a dict with ``cells``/``sites``/``n_steps``.

    Cells are returned in ``row`` order so they align with ``acts[row]``.
    """
    idx = json.loads((dump / "activations" / "index.json").read_text())
    cells = idx["cells"] if isinstance(idx, dict) else idx
    if cells and "row" in cells[0]:
        cells = sorted(cells, key=lambda c: int(c["row"]))
        if [int(c["row"]) for c in cells] != list(range(len(cells))):
            raise ValueError("index rows are not 0..n-1")
    return cells


def load_index_meta(dump: Path) -> dict[str, Any]:
    idx = json.loads((dump / "activations" / "index.json").read_text())
    return {k: v for k, v in idx.items() if k != "cells"} if isinstance(idx, dict) else {}


def list_sites(dump: Path) -> list[str]:
    return sorted(p.stem for p in (dump / "activations").glob("*.npz"))


def load_site_features(dump: Path, site_id: str, step: int, token_pool: str, token_sets=None, cells=None) -> np.ndarray:
    """Reduce [n, T, N, d] -> [n, f] per the requested imagined step and token pooling.

    Group-restricted dumps carry ``token_index`` [n, T, N] with the spatial id of
    each stored token (-1 = padding); ``mean`` then averages only real tokens and
    ``group:<name>`` averages the tokens of that token group (egg, gripper,
    gripper_corridor, ...) using the per-scene sets from ``token_sets``.
    ``token:<i>`` / ``tokens:<lo>-<hi>`` refer to spatial ids when ``token_index``
    is present, else to positions.
    """
    with np.load(dump / "activations" / f"{site_id}.npz") as z:
        acts = np.asarray(z["acts"], dtype=np.float32)
        tix = np.asarray(z["token_index"]) if "token_index" in z else None
    if acts.ndim != 4:
        raise ValueError(f"{site_id}: expected acts [n, T, N, d], got {acts.shape}")
    T = acts.shape[1]
    t = step if step >= 0 else T + step
    if not 0 <= t < T:
        raise ValueError(f"{site_id}: step {step} out of range for T={T}")
    a = acts[:, t]  # [n, N, d]
    valid = np.ones(a.shape[:2], dtype=bool) if tix is None else (tix[:, t] >= 0)
    if a.shape[1] == 1:  # AdaLN-style single vector
        return a[:, 0]

    def masked_mean(mask):
        out = np.full((a.shape[0], a.shape[2]), np.nan, dtype=np.float32)
        for i in range(a.shape[0]):
            if mask[i].any():
                out[i] = a[i, mask[i]].mean(axis=0)
        return out

    if token_pool == "mean":
        return masked_mean(valid)
    if token_pool == "flatten":
        return np.where(valid[:, :, None], a, 0.0).reshape(a.shape[0], -1)
    if token_pool.startswith("group:"):
        if tix is None or token_sets is None or cells is None:
            raise ValueError("group:<name> pooling needs token_index in the dump and token_sets from the stimulus masks")
        g = token_pool.split(":", 1)[1]
        mask = np.zeros_like(valid)
        for i, c in enumerate(cells):
            mask[i] = valid[i] & np.isin(tix[i, t], token_sets[(c["pair_id"], t, g)])
        return masked_mean(mask)
    if token_pool.startswith("pair:"):
        # concatenated [z_g | z_e] group means for the bilinear entrant (and linear-on-concatenation baselines)
        if tix is None or token_sets is None or cells is None:
            raise ValueError("pair:<g>,<e> pooling needs token_index in the dump and token_sets from the stimulus masks")
        g1, g2 = token_pool.split(":", 1)[1].split(",")
        parts = []
        for g in (g1, g2):
            mask = np.zeros_like(valid)
            for i, c in enumerate(cells):
                mask[i] = valid[i] & np.isin(tix[i, t], token_sets[(c["pair_id"], t, g)])
            parts.append(masked_mean(mask))
        return np.concatenate(parts, axis=1)
    if token_pool.startswith("token:"):
        i = int(token_pool.split(":", 1)[1])
        if tix is None:
            return a[:, i]
        return masked_mean(valid & (tix[:, t] == i))
    if token_pool.startswith("tokens:"):
        lo, hi = (int(v) for v in token_pool.split(":", 1)[1].split("-"))
        if tix is None:
            return a[:, lo:hi].mean(axis=1)
        return masked_mean(valid & (tix[:, t] >= lo) & (tix[:, t] < hi))
    raise ValueError(f"unknown token_pool {token_pool}")


def load_manifest(stimulus: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(l) for l in (stimulus / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return {r["cell_id"]: r for r in rows}


def load_summary_angles(stimulus: Path) -> dict[str, float]:
    """Hazard bearing per pair from summary.json: control-pose angle for H0, 0 (on-path) for H1."""
    path = stimulus / "summary.json"
    if not path.exists():
        return {}
    s = json.loads(path.read_text())
    out = {}
    for p in s.get("pairs", []):
        sel = p.get("control_pose_selection") or {}
        if "angle_rad" in sel:
            out[p["pair_id"]] = float(sel["angle_rad"])
    return out


def build_labels_and_meta(
    index: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    stimulus: Path,
    angle_field: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Return (labels, meta, nuisance) aligned with the index order."""
    n = len(index)
    pair = np.asarray([c["pair_id"] for c in index])
    hz = np.asarray([int(c["hazard"]) for c in index])
    ac = np.asarray([int(c["candidate_action"]) for c in index])
    seed = np.asarray([int(c.get("seed", -1)) for c in index])
    labels: dict[str, np.ndarray] = {}
    for name, (fld, kind) in TARGETS.items():
        if fld is None:
            continue
        vals = np.asarray([float(manifest[c["cell_id"]].get(fld, np.nan)) for c in index])
        labels[name] = (vals > 0).astype(float) if kind == "binary" else vals
    labels["interaction"] = interaction_target(labels["contact"], labels["force"], pair, hz, ac)
    families = np.asarray([str(manifest[c["cell_id"]].get("risk_family", "unknown")) for c in index])

    pair_angles = load_summary_angles(stimulus)
    angle = np.zeros(n)
    proprio = []
    act_mag = np.zeros(n)
    centroid = np.zeros((n, 2))
    visible = np.zeros(n)
    for i, c in enumerate(index):
        row = manifest[c["cell_id"]]
        cen = row.get("target_centroid_xy") or [np.nan, np.nan]
        centroid[i] = cen
        visible[i] = float(row.get("target_visible_pixels", np.nan))
        cell_path = stimulus / row["artifact"]
        pro = np.zeros(7)
        if cell_path.exists():
            with np.load(cell_path) as z:
                if "model_actions" in z:
                    act_mag[i] = float(np.linalg.norm(np.asarray(z["model_actions"], dtype=np.float64)))
                if "context_proprios" in z:
                    pro = np.asarray(z["context_proprios"], dtype=np.float64)[-1]
                if angle_field and angle_field in z:
                    angle[i] = float(np.asarray(z[angle_field]).ravel()[0])
        proprio.append(pro)
        if not angle_field:
            # bearing of the hazard relative to the approach path: on-path (H1) = 0,
            # off-path control pose angle (H0) from summary.json
            angle[i] = 0.0 if hz[i] == 1 else pair_angles.get(c["pair_id"], np.pi)
    proprio_arr = np.asarray(proprio)
    # egg patch-token position (row, col of the 16 px patch holding the egg centroid) as an ordinal nuisance
    token_rc = np.floor(np.nan_to_num(centroid)[:, ::-1] / PATCH_PX)
    nuisance = np.column_stack([proprio_arr, act_mag, np.nan_to_num(centroid), token_rc, np.nan_to_num(visible)])
    meta = {"pair_id": pair, "hazard": hz, "action": ac, "seed": seed, "angle": angle, "family": families}
    return labels, meta, nuisance


NUISANCE_COLUMNS = ["proprio_0..6 (eef xyz, euler, gripper)", "action_chunk_l2", "egg_centroid_x", "egg_centroid_y",
                    "egg_token_row", "egg_token_col", "egg_visible_px"]


def interaction_target(contact: np.ndarray, force: np.ndarray, pair: np.ndarray, hz: np.ndarray, ac: np.ndarray) -> np.ndarray:
    """Per-cell interaction label: contrast (2h-1)(2a-1) signed by the scene's contact DiD (force DiD as fallback)."""
    y = np.full(len(pair), np.nan)
    for p in np.unique(pair):
        idx = np.flatnonzero(pair == p)
        cell = {(int(hz[i]), int(ac[i])): i for i in idx}
        if not all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
            continue

        def did(v):
            return v[cell[(1, 1)]] - v[cell[(1, 0)]] - v[cell[(0, 1)]] + v[cell[(0, 0)]]

        s = did(contact)
        if not np.isfinite(s) or abs(s) < 1e-12:
            s = did(force)
        if not np.isfinite(s) or abs(s) < 1e-12:
            continue
        for i in idx:
            y[i] = float(np.sign(s) * (2 * hz[i] - 1) * (2 * ac[i] - 1) > 0)
    return y


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def metric_for(kind: str, scores: np.ndarray, y: np.ndarray) -> float:
    return auroc(scores, y) if kind == "binary" else r_squared(scores, y)


def cross_fit_scores(
    make_model, X: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray], n_folds: int, seed: int
) -> np.ndarray:
    """Out-of-fold scores on the training set (scene folds) so the nuisance-incremental fit is not leaked."""
    out = np.full(len(y), np.nan)
    for held in scene_folds(meta["pair_id"], n_folds, seed):
        train = np.setdiff1d(np.arange(len(y)), held)
        if len(np.unique(meta["pair_id"][train])) < 2:
            continue
        m = make_model().fit(X[train], y[train], {k: v[train] for k, v in meta.items()})
        out[held] = m.score(X[held])
    return out


def within_scene_interaction_agreement(scores: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray]) -> dict:
    """Fraction of scenes where the readout's DiD sign matches the label's DiD sign (nonzero label DiD only)."""
    pair, hz, ac = meta["pair_id"], meta["hazard"], meta["action"]
    agree, total = 0, 0
    for p in np.unique(pair):
        idx = np.flatnonzero(pair == p)
        cell = {(int(hz[i]), int(ac[i])): i for i in idx}
        if not all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
            continue

        def did(v):
            return v[cell[(1, 1)]] - v[cell[(1, 0)]] - v[cell[(0, 1)]] + v[cell[(0, 0)]]

        dy = did(y)
        if abs(dy) < 1e-12 or not np.isfinite(did(scores)):
            continue
        total += 1
        agree += int(np.sign(did(scores)) == np.sign(dy))
    return {"n_scenes": total, "sign_agreement": (agree / total) if total else float("nan")}


def incremental_over_nuisance(
    kind: str,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    N_train: np.ndarray,
    N_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    clusters_test: np.ndarray,
    lam: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ok = np.isfinite(train_scores)
    Ns = (N_train - N_train.mean(0)) / (N_train.std(0) + 1e-9)
    Nt = (N_test - N_train.mean(0)) / (N_train.std(0) + 1e-9)
    w0, b0 = ridge_fit(Ns[ok], y_train[ok], lam)
    base_pred = ridge_predict(Nt, w0, b0)
    Xa = np.column_stack([Ns[ok], train_scores[ok]])
    w1, b1 = ridge_fit(Xa, y_train[ok], lam)
    full_pred = ridge_predict(np.column_stack([Nt, test_scores]), w1, b1)

    def delta(idx):
        return metric_for(kind, full_pred[idx], y_test[idx]) - metric_for(kind, base_pred[idx], y_test[idx])

    boot = cluster_bootstrap(delta, clusters_test, n_boot=n_boot, seed=seed)
    return {
        "nuisance_only": metric_for(kind, base_pred, y_test),
        "nuisance_plus_model": metric_for(kind, full_pred, y_test),
        "delta": boot,
        "significant": bool(np.isfinite(boot["ci_low"]) and boot["ci_low"] > 0.0),
    }


def stability(make_model, X, y, meta, seeds: list[int], frac: float = 0.8) -> dict[str, Any]:
    """Refit on three scene-resamples of the discovery set and report pairwise chart agreement + rank."""
    pair = meta["pair_id"]
    uniq = np.unique(pair)
    fits, ranks = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        keep = rng.choice(uniq, size=max(2, int(round(frac * len(uniq)))), replace=False)
        idx = np.flatnonzero(np.isin(pair, keep))
        try:
            m = make_model().fit(X[idx], y[idx], {k: v[idx] for k, v in meta.items()})
        except Exception as exc:  # e.g. missing angle / factorial cells
            return {"error": repr(exc)}
        fits.append(m)
        g = m.geometry_summary()
        ranks.append(g.get("rank"))
    sims = [fits[i].similarity(fits[j]) for i in range(len(fits)) for j in range(i + 1, len(fits))]
    sims = [s for s in sims if np.isfinite(s)]
    return {
        "seeds": seeds,
        "pairwise_similarity": sims,
        "mean_similarity": float(np.mean(sims)) if sims else float("nan"),
        "ranks": ranks,
        "rank_stable": bool(len(set(ranks)) == 1),
    }


def evaluate_site(
    site_id: str,
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    nuisance: np.ndarray,
    disc: np.ndarray,
    conf: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"site_id": site_id, "n_features": int(X.shape[1]), "models": {}}
    Xd = X[disc]
    md = {k: v[disc] for k, v in meta.items()}
    result["interaction_effective_rank"] = int(
        effective_rank(JacobianLocal.within_scene_differences(Xd, md))
        if len(np.unique(md["pair_id"])) >= 2
        else 0
    )
    families = np.unique(meta["family"])
    stats = clean_distribution(Xd)
    edits_by_model: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if args.frames is not None:
        try:
            result["frames"] = frame_table(
                Xd, {k: v[disc] for k, v in args.frames["coords"].items()}, md,
                budget=args.budget, lam=args.lam, n_folds=args.n_folds, n_boot=args.n_boot, seed=args.split_seed,
                scene_centered=not args.no_scene_center,
            )
            result["frames"]["dropped_frames"] = args.frames["dropped_frames"]
            result["frames"]["reprojection_error_px"] = args.frames["reprojection_error_px"]
        except Exception as exc:
            result["frames"] = {"error": repr(exc)}
    for name in args.models:
        maker = lambda name=name: MODELS[name](budget=args.budget, lam=args.lam)
        entry: dict[str, Any] = {"budget": args.budget if name not in ("linear_meandiff", "linear_ridge", "counterfactual_pair_direction") else 1}
        for tname in args.targets:
            kind = TARGETS[tname][1]
            y_all = labels[tname]
            # the interaction contrast is an XOR of the main effects: residualize the action main effect
            # (and scene mean) out of the features so only the interaction component can separate cells
            X_t = quartet_residualize(X, meta, keep_hazard=True) if tname == "interaction" else X
            valid = np.isfinite(y_all) & np.all(np.isfinite(X_t), axis=1)
            d_idx = disc[valid[disc]]
            c_idx = conf[valid[conf]]
            tres: dict[str, Any] = {"n_discovery_cells": int(len(d_idx)), "n_confirmation_cells": int(len(c_idx))}
            try:
                if len(np.unique(meta["pair_id"][d_idx])) < 2:
                    raise ValueError("fewer than two discovery scenes with a finite label")
                Xdt, ydt, mdt = X_t[d_idx], y_all[d_idx], {k: v[d_idx] for k, v in meta.items()}
                # discovery: scene-fold CV (model selection numbers)
                cv_scores = cross_fit_scores(maker, Xdt, ydt, mdt, args.n_folds, args.split_seed)
                ok = np.isfinite(cv_scores)
                tres["discovery_cv"] = cluster_bootstrap(
                    lambda idx: metric_for(kind, cv_scores[ok][idx], ydt[ok][idx]),
                    mdt["pair_id"][ok],
                    n_boot=args.n_boot,
                    seed=args.split_seed,
                )
                tres["discovery_within_scene"] = within_scene_interaction_agreement(cv_scores, ydt, mdt)
                # confirmation: one fit on all discovery, scored once on confirmation
                model = maker().fit(Xdt, ydt, mdt)
                if len(c_idx):
                    cs = model.score(X_t[c_idx])
                    yct = y_all[c_idx]
                    mc = {k: v[c_idx] for k, v in meta.items()}
                    tres["confirmation"] = cluster_bootstrap(
                        lambda idx: metric_for(kind, cs[idx], yct[idx]),
                        mc["pair_id"],
                        n_boot=args.n_boot,
                        seed=args.split_seed,
                    )
                    tres["confirmation_within_scene"] = within_scene_interaction_agreement(cs, yct, mc)
                    tres["incremental_over_nuisance"] = incremental_over_nuisance(
                        kind, cv_scores, cs, nuisance[d_idx], nuisance[c_idx], ydt, yct,
                        mc["pair_id"], args.lam, args.n_boot, args.split_seed,
                    )
                else:
                    tres["confirmation"] = None
                tres["geometry"] = model.geometry_summary()
                tres["stability"] = stability(maker, Xdt, ydt, mdt, args.stability_seeds)
                if name == "lowrank_subspace":
                    tres["rank_sweep"] = rank_sweep(kind, Xdt, ydt, mdt, args)
                # transport across risk families (fit on one, score on the others)
                if len(families) > 1:
                    tr = {}
                    for fam in families:
                        src = np.flatnonzero(mdt["family"] == fam)
                        dst = np.flatnonzero((meta["family"] != fam) & valid)
                        if len(np.unique(mdt["pair_id"][src])) < 2 or len(dst) == 0:
                            continue
                        m = maker().fit(Xdt[src], ydt[src], {k: v[src] for k, v in mdt.items()})
                        tr[str(fam)] = metric_for(kind, m.score(X_t[dst]), y_all[dst])
                    tres["transport"] = {"status": "tested", "by_source_family": tr}
                else:
                    tres["transport"] = {"status": "task_local", "note": "single risk_family; transport untested"}
                # gate verdict on confirmation if present else discovery CV
                headline = tres["confirmation"] or tres["discovery_cv"]
                thr = GATES["auroc"] if kind == "binary" else GATES["r2"]
                stab = tres["stability"].get("mean_similarity", float("nan"))
                tres["gate"] = {
                    "metric_threshold": thr,
                    "metric_pass": bool(np.isfinite(headline["point"]) and headline["point"] >= thr),
                    "incremental_pass": bool(tres.get("incremental_over_nuisance", {}).get("significant", False)),
                    "stability_pass": bool(np.isfinite(stab) and stab >= GATES["stability"]),
                    "transport_pass": None if tres["transport"]["status"] == "task_local" else bool(
                        sum(v >= thr for v in tres["transport"]["by_source_family"].values()) >= 2
                    ),
                }
                # edit export for the confirmation cells (or discovery if none)
                cells_idx = c_idx if len(c_idx) else d_idx
                if args.export_edits:
                    ys = y_all[cells_idx]
                    target_val = float(ys.max()) if kind == "binary" else float(np.nanpercentile(ys, 90))
                    norm = args.edit_norm_frac * float(np.linalg.norm(Xd, axis=1).mean())
                    deltas = np.stack([model.edit_direction(X_t[i], target_val, norm) for i in cells_idx])
                    cell_ids = [str(args.index[i]["cell_id"]) for i in cells_idx]
                    om = on_manifold_scores(X_t[cells_idx], deltas, stats)
                    om_random = on_manifold_scores(X_t[cells_idx], random_equal_norm_edits(deltas, args.split_seed), stats)
                    p = write_edit_artifact(
                        out_dir / "edits" / f"{name}__{site_id}__{tname}",
                        model_name=name, site_id=site_id, step=args.step, token_pool=args.token_pool,
                        cell_ids=cell_ids, deltas=deltas, norm=norm, target=target_val, target_name=tname,
                        extra={"on_manifold": om, "on_manifold_random_equal_norm": om_random},
                    )
                    tres["edit_artifact"] = str(p)
                    tres["on_manifold"] = om
                    tres["on_manifold_random_equal_norm"] = om_random
                    edits_by_model[(name, tname)] = (cells_idx, deltas)
            except Exception as exc:
                tres["error"] = repr(exc)
            entry[tname] = tres
        result["models"][name] = entry
    # compare every model's edits to the counterfactual-pair reference on cosine
    result["edit_comparison"] = {}
    for tname in args.targets:
        ref = edits_by_model.get(("counterfactual_pair_direction", tname))
        if ref is None:
            continue
        comp = {}
        for (name, tn), (idx, deltas) in edits_by_model.items():
            if tn != tname or name == "counterfactual_pair_direction":
                continue
            if len(idx) == len(ref[0]) and np.array_equal(idx, ref[0]):
                comp[name] = {
                    "cosine_to_counterfactual": edit_cosine(deltas, ref[1]),
                    "on_manifold_increase": result["models"][name][tname]["on_manifold"]["increase_mean"],
                }
        comp["counterfactual_pair_direction"] = {
            "cosine_to_counterfactual": 1.0,
            "on_manifold_increase": result["models"]["counterfactual_pair_direction"][tname]["on_manifold"]["increase_mean"],
        }
        result["edit_comparison"][tname] = comp
    # which model best predicts each target (headline metric), interaction included
    best = {}
    for tname in args.targets:
        cands = []
        for name, entry in result["models"].items():
            tr = entry.get(tname, {})
            head = tr.get("confirmation") or tr.get("discovery_cv")
            if head and np.isfinite(head.get("point", np.nan)):
                cands.append((head["point"], name))
        if cands:
            best[tname] = {"model": max(cands)[1], "metric": max(cands)[0]}
    result["best_model_by_target"] = best
    return result


def rank_sweep(kind: str, X: np.ndarray, y: np.ndarray, meta: dict[str, np.ndarray], args) -> dict[str, Any]:
    """Smallest low-rank budget whose discovery-CV metric reaches the gate (CQM-style bottleneck check)."""
    thr = GATES["auroc"] if kind == "binary" else GATES["r2"]
    kmax = int(min(X.shape[1], max(2, len(y) - 2)))
    curve = []
    needed = None
    for k in [k for k in RANK_SWEEP if k <= kmax]:
        s = cross_fit_scores(lambda k=k: LowRankSubspace(budget=k, lam=args.lam), X, y, meta, args.n_folds, args.split_seed)
        ok = np.isfinite(s)
        m = metric_for(kind, s[ok], y[ok])
        curve.append({"rank": k, "metric": m})
        if needed is None and np.isfinite(m) and m >= thr:
            needed = k
    return {"gate": thr, "curve": curve, "rank_needed_for_gate": needed, "max_rank_tested": kmax}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def read_seeds(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    return {int(v) for v in path.read_text().split() if v.strip()}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sites", nargs="*", default=None, help="site ids (default: all in dump)")
    ap.add_argument("--models", nargs="*", default=None, help="default: all entrants; 'bilinear' only with --token-pool pair:<g>,<e>")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--discovery-seeds", type=Path, default=None)
    ap.add_argument("--confirmation-seeds", type=Path, default=None)
    ap.add_argument("--allow-calibration", action="store_true", help="permit seeds 101/102 (smoke tests only)")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=-1, help="imagined step index (negative from end)")
    ap.add_argument("--token-pool", default="mean")
    ap.add_argument("--angle-field", default=None, help="cell npz key holding the circular variable; default: bearing from summary")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--stability-seeds", type=int, nargs=3, default=[0, 1, 2])
    ap.add_argument("--no-frames", action="store_true", help="skip the coordinate-frame triangulation stage")
    ap.add_argument("--no-scene-center", action="store_true",
                    help="disable within-scene centering of features (default: subtract each scene's cell mean)")
    ap.add_argument("--export-edits", action="store_true")
    ap.add_argument("--edit-norm-frac", type=float, default=0.1, help="edit norm as a fraction of mean activation norm")
    args = ap.parse_args(argv)
    if args.models is None:
        args.models = [m for m in MODELS if m != "bilinear" or args.token_pool.startswith("pair:")]

    index = load_index(args.dump)
    args.index = index
    manifest = load_manifest(args.stimulus)
    missing = [c["cell_id"] for c in index if c["cell_id"] not in manifest]
    if missing:
        raise SystemExit(f"cells in dump missing from manifest: {missing[:5]}")
    labels, meta, nuisance = build_labels_and_meta(index, manifest, args.stimulus, args.angle_field)
    args.frames = None
    if not args.no_frames:
        try:
            args.frames = collect_frames(args.stimulus, [manifest[c["cell_id"]] for c in index])
            if args.angle_field is None:
                # path-relative bearing of the hazard is the circular variable
                meta["angle"] = np.asarray(args.frames["extras"]["bearing"], dtype=np.float64)
                meta["angle_source"] = np.asarray(["path_bearing"] * len(index))
        except Exception as exc:
            args.frames = {"error": repr(exc)}
            print(f"[frames] skipped: {exc!r}", file=sys.stderr)
            args.frames = None

    seeds = meta["seed"]
    disc_set = read_seeds(args.discovery_seeds)
    conf_set = read_seeds(args.confirmation_seeds) or set()
    if disc_set is None:
        disc_set = set(int(s) for s in np.unique(seeds)) - conf_set
    if not args.allow_calibration:
        bad = (disc_set | conf_set) & CALIBRATION_SEEDS
        if bad:
            raise SystemExit(f"calibration seeds {sorted(bad)} may not be used; pass --allow-calibration for smoke tests")
    if disc_set & conf_set:
        raise SystemExit("discovery and confirmation seed sets overlap")
    disc = np.flatnonzero(np.isin(seeds, sorted(disc_set)))
    conf = np.flatnonzero(np.isin(seeds, sorted(conf_set)))
    if len(np.unique(meta["pair_id"][disc])) < 2:
        raise SystemExit("need at least two discovery scenes")

    sites = args.sites or list_sites(args.dump)
    args.out.mkdir(parents=True, exist_ok=True)
    index_meta = load_index_meta(args.dump)
    token_sets = None
    if args.token_pool.startswith("group:") or args.token_pool.startswith("pair:"):
        from geometry_localize import group_token_sets  # noqa: E402  (numpy-only)
        n_steps = int(index_meta.get("n_steps") or 3)
        gnames = tuple(args.token_pool.split(":", 1)[1].split(","))
        token_sets = group_token_sets(args.stimulus, [dict(c, artifact=c.get("artifact", "")) for c in index], gnames, n_steps, int(index_meta.get("group_frame_offset", 0)))
    results = []
    for site in sites:
        X = load_site_features(args.dump, site, args.step, args.token_pool, token_sets, index)
        bad = ~np.all(np.isfinite(X), axis=1)
        if bad.any():
            # cells with no token in the requested pool: fill with the scene mean of the others (rare)
            for i in np.flatnonzero(bad):
                same = (meta["pair_id"] == meta["pair_id"][i]) & ~bad
                X[i] = X[same].mean(axis=0) if same.any() else np.nan_to_num(X[~bad].mean(axis=0))
        if not args.no_scene_center:
            X = scene_center(X, meta["pair_id"])
        results.append(evaluate_site(site, X, labels, meta, nuisance, disc, conf, args, args.out))

    report = {
        "protocol": "cgs-geometry-tournament-v0.1",
        "dump": str(args.dump),
        "stimulus": str(args.stimulus),
        "step": args.step,
        "token_pool": args.token_pool,
        "scene_centered": not args.no_scene_center,
        "frames_stage": "skipped" if args.frames is None else "ran",
        "dump_index_meta": {k: v for k, v in index_meta.items() if k in ("n_steps", "dump_groups", "group_frame_offset", "layout", "dtype")},
        "angle_source": "path_bearing" if (args.frames is not None and args.angle_field is None) else (args.angle_field or "summary_control_angle"),
        "nuisance_columns": NUISANCE_COLUMNS,
        "rank_sweep": list(RANK_SWEEP),
        "budget": args.budget,
        "lam": args.lam,
        "gates": GATES,
        "n_cells": len(index),
        "discovery_seeds": sorted(disc_set),
        "confirmation_seeds": sorted(conf_set),
        "n_discovery_scenes": int(len(np.unique(meta["pair_id"][disc]))),
        "n_confirmation_scenes": int(len(np.unique(meta["pair_id"][conf]))),
        "families": sorted(set(meta["family"].tolist())),
        "transcoder": "placeholder; gated behind causal localization (design Phase 3)",
        "sites": results,
    }
    (args.out / "tournament.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    # compact stdout summary
    for r in results:
        for mname, entry in r["models"].items():
            for tname in args.targets:
                t = entry.get(tname, {})
                head = t.get("confirmation") or t.get("discovery_cv") or {}
                print(f"{r['site_id']:>28s} {mname:>18s} {tname:>9s} metric={head.get('point', float('nan')):.3f} "
                      f"gate={t.get('gate', {}).get('metric_pass')} err={t.get('error', '')}")
    return report


if __name__ == "__main__":
    main()
