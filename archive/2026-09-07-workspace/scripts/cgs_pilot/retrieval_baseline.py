#!/usr/bin/env python3
"""Retrieval / copying baselines for the hazard x action INTERACTION.

The question is not whether the predictor is accurate but whether the
action-conditional hazard interaction it predicts could come from retrieval or
scene-blind action response. Each baseline produces, per scene, a predicted
interaction

    I_pred = Y(H1,A1) - Y(H1,A0) - Y(H0,A1) + Y(H0,A0)

in DINOv3 latent space (last imagined step, egg + gripper tokens when masks exist)
which is compared with I_true from the true future latents by cosine.

Baselines:
  (a) ``knn_droid``: k-NN over DROID (frame latent, action chunk) transitions,
      predicting each cell's latent delta by copy-delta from the retrieved DROID
      transitions (``droid_reference.py`` + ``latent_cache.py --droid-reference``);
      marked deferred when the DROID latents are absent.
  (b) ``loso_ridge``: leave-one-scene-out ridge regression from the action chunk
      to the latent delta, fit on the other scenes (scene-blind action response).
  (c) ``persistence``: Y = context (I_pred = 0 by construction, cosine 0).
  (d) ``copy_delta_scene``: nearest other-scene cell by (context latent, action),
      Y = own context + donor delta.

Gate: the model's cross-scene mean cosine(I_pred, I_true) must exceed every
baseline's, with a sign-flip test on per-scene differences p < 0.05. All
inference is at scene level (pair_id).

``sonar`` block: the same comparison after projecting every interaction vector
onto the leave-one-scene-out top-k subspace of the OTHER scenes' true
interactions (pooled and per-token features; see ``sonar_metrics.py``,
arXiv:2510.00845 for the cross-validation logic). The subspace is built from
TRUE futures only, so it favours no predictor over another.

Related: interaction-level evaluation of counterfactual world models
(arXiv:2608.22092); action-conditioned latent evaluation with region masks
(arXiv:2608.11601).
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
from protocol import CELL_ORDER, OBJECT_HAZARD, canonical_json, sha256_file  # noqa: E402
from token_groups import DOMAINS  # noqa: E402

REGION_LABEL = {"egg": "egg|gripper", "driving": "hazard|corridor"}
from sonar_metrics import project, svd_subspace  # noqa: E402
from stats_utils import cluster_bootstrap_mean, cosine, finite, sign_flip_p  # noqa: E402


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #


def group_cells(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], int]]:
    grouped: dict[str, dict[tuple[int, int], int]] = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["pair_id"], {})[(int(r["hazard"]), int(r["candidate_action"]))] = i
    return {p: c for p, c in grouped.items() if set(CELL_ORDER) <= set(c)}


def interaction(y: np.ndarray, cells: dict[tuple[int, int], int], level: int = 1) -> np.ndarray:
    return y[cells[(level, 1)]] - y[cells[(level, 0)]] - y[cells[(0, 1)]] + y[cells[(0, 0)]]


def identity_cells(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], int]]:
    """Scenes carrying the quartet AND the level-3 (object on path) pair (driving, v0.8)."""

    grouped: dict[str, dict[tuple[int, int], int]] = {}
    for i, r in enumerate(rows):
        grouped.setdefault(r["pair_id"], {})[(int(r["hazard"]), int(r["candidate_action"]))] = i
    need = set(CELL_ORDER) | {(OBJECT_HAZARD, 0), (OBJECT_HAZARD, 1)}
    return {p: c for p, c in grouped.items() if need <= set(c)}


def identity_block(preds: dict[str, np.ndarray | None], target: np.ndarray, groups3: dict, masks: np.ndarray) -> dict[str, Any]:
    """Model vs baselines on the object interaction (level 3 vs 0) and on the identity contrast I_ped - I_obj."""

    def cosines(y: np.ndarray, kind: str) -> dict[str, float]:
        out = {}
        for pid, cells in groups3.items():
            m = scene_mask(masks, cells)
            if kind == "object":
                a, b = interaction(y, cells, OBJECT_HAZARD), interaction(target, cells, OBJECT_HAZARD)
            else:
                a = interaction(y, cells, 1) - interaction(y, cells, OBJECT_HAZARD)
                b = interaction(target, cells, 1) - interaction(target, cells, OBJECT_HAZARD)
            out[pid] = masked_cosine(a, b, m)
        return out

    block: dict[str, Any] = {"n_scenes": len(groups3)}
    for kind in ("object", "identity"):
        model_cos = cosines(preds["model"], kind)
        entry: dict[str, Any] = {
            "model": {"per_scene_cosine": model_cos, "mean_cosine": cluster_bootstrap_mean(np.array(list(model_cos.values()))),
                      "sign_flip_p_vs_zero": sign_flip_p(np.array(list(model_cos.values())))},
            "baselines": {},
        }
        for name, y in preds.items():
            if name == "model":
                continue
            if y is None:
                entry["baselines"][name] = {"status": "deferred"}
                continue
            bc = cosines(y, kind)
            entry["baselines"][name] = {"per_scene_cosine": bc, **compare_to_model(model_cos, bc)}
        block[kind] = entry
    block["definition"] = "object: I_obj = Y(3,1) - Y(3,0) - Y(0,1) + Y(0,0) vs true; identity: I_ped - I_obj vs true (v0.8 identity x action contrast)"
    return block


def scene_mask(mask: np.ndarray, cells: dict[tuple[int, int], int]) -> np.ndarray:
    """Union of the four cells' token masks -> bool [16, 16]."""

    return np.any(np.stack([mask[cells[k]] for k in CELL_ORDER]), axis=0)


def masked_cosine(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    return cosine(a[m], b[m])


def per_scene_cosines(y_pred: np.ndarray, y_true: np.ndarray, groups: dict, masks: np.ndarray) -> dict[str, float]:
    out = {}
    for pair_id, cells in groups.items():
        m = scene_mask(masks, cells)
        out[pair_id] = masked_cosine(interaction(y_pred, cells), interaction(y_true, cells), m)
    return out


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def combined_distance(ctx_a: np.ndarray, act_a: np.ndarray, ctx_b: np.ndarray, act_b: np.ndarray, action_weight: float) -> np.ndarray:
    dc = pairwise_l2(ctx_a.reshape(len(ctx_a), -1), ctx_b.reshape(len(ctx_b), -1))
    da = pairwise_l2(act_a.reshape(len(act_a), -1), act_b.reshape(len(act_b), -1))
    sc = np.median(dc[dc > 0]) if (dc > 0).any() else 1.0
    sa = np.median(da[da > 0]) if (da > 0).any() else 1.0
    return dc / max(sc, 1e-12) + action_weight * da / max(sa, 1e-12)


def copy_delta_scene(context: np.ndarray, target: np.ndarray, pooled_ctx: np.ndarray, actions: np.ndarray, scene_ids: list[str], action_weight: float) -> tuple[np.ndarray, np.ndarray]:
    d = combined_distance(pooled_ctx, actions, pooled_ctx, actions, action_weight)
    s = np.asarray(scene_ids)
    d[s[:, None] == s[None, :]] = np.inf
    donors = d.argmin(axis=1)
    return context + (target[donors] - context[donors]), donors


def knn_droid(context: np.ndarray, pooled_ctx: np.ndarray, actions: np.ndarray, droid_pooled: np.ndarray, droid_delta: np.ndarray, droid_actions: np.ndarray, k: int, action_weight: float) -> np.ndarray:
    d = combined_distance(pooled_ctx, actions, droid_pooled, droid_actions, action_weight)
    idx = np.argsort(d, axis=1)[:, :k]
    delta = np.stack([droid_delta[i].astype(np.float32).mean(0) for i in idx])
    return context + delta


def loso_ridge(context: np.ndarray, target: np.ndarray, actions: np.ndarray, scene_ids: list[str], lam: float) -> np.ndarray:
    """Predict delta = f(action chunk) with ridge fit on other scenes only."""

    x = actions.reshape(len(actions), -1).astype(np.float64)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    delta = (target - context).reshape(len(context), -1).astype(np.float64)
    s = np.asarray(scene_ids)
    pred = np.zeros_like(delta)
    for scene in np.unique(s):
        tr = s != scene
        xt = x[tr]
        beta = np.linalg.solve(xt.T @ xt + lam * np.eye(x.shape[1]), xt.T @ delta[tr])
        pred[~tr] = x[~tr] @ beta
    return context + pred.reshape(context.shape).astype(np.float32)


def projected_scene_cosines(preds: dict[str, np.ndarray | None], target: np.ndarray, groups: dict, masks: np.ndarray, ks: tuple[int, ...] = (1, 2, 4, 8)) -> dict[str, Any]:
    """LOSO projected cosine(I_pred, I_true) for the model and every baseline."""

    ids = sorted(groups)
    if len(ids) < 3:
        return {"status": "insufficient_scenes"}
    tok = {p: np.where(scene_mask(masks, groups[p]).ravel())[0] for p in ids}

    def vec(y, p):
        return interaction(y.reshape(len(y), -1, y.shape[-1]), groups[p])[tok[p]].astype(np.float64)

    i_true = {p: vec(target, p) for p in ids}
    out: dict[str, Any] = {}
    for feature in ("pooled", "per_token"):
        rep = (lambda v: v.mean(axis=0, keepdims=True)) if feature == "pooled" else (lambda v: v)
        out[feature] = {}
        for k in ks:
            per: dict[str, dict[str, float]] = {name: {} for name in preds}
            for p in ids:
                X = np.concatenate([rep(i_true[o]) for o in ids if o != p], axis=0)
                Q = svd_subspace(X, min(k, len(X) - 1) if feature == "pooled" else k)
                pt = project(rep(i_true[p]), Q)
                for name, y in preds.items():
                    if y is None:
                        continue
                    per[name][p] = cosine(project(rep(vec(y, p)), Q), pt)
            block = {}
            for name, d in per.items():
                if not d:
                    block[name] = {"status": "deferred"}
                    continue
                vals = np.array([d[p] for p in ids])
                entry = {**cluster_bootstrap_mean(vals), "sign_flip_p_vs_zero": sign_flip_p(vals)}
                if name != "model" and per["model"]:
                    diff = np.array([per["model"][p] - d[p] for p in ids])
                    entry["model_minus_baseline"] = {**cluster_bootstrap_mean(diff), "sign_flip_p": sign_flip_p(diff)}
                block[name] = entry
            out[feature][f"k{k}"] = block
    return out


def compare_to_model(model_cos: dict[str, float], base_cos: dict[str, float]) -> dict[str, Any]:
    pairs = sorted(set(model_cos) & set(base_cos))
    diff = np.array([model_cos[p] - base_cos[p] for p in pairs])
    return {
        "model_mean_cosine": float(np.mean([model_cos[p] for p in pairs])) if pairs else float("nan"),
        "baseline_mean_cosine": float(np.mean([base_cos[p] for p in pairs])) if pairs else float("nan"),
        "diff": cluster_bootstrap_mean(diff),
        "sign_flip_p": sign_flip_p(diff),
        "n_scenes": len(pairs),
    }


def analyze(cache: dict[str, Any], rows: list[dict[str, Any]], action_weight: float, k: int, lam: float, domain: str = "egg") -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain}")
    context = cache["context"].astype(np.float32)  # [n, 16, 16, D]
    target = cache["target"][:, -1].astype(np.float32)
    model_pred = cache["prediction"][:, -1].astype(np.float32)
    actions = cache["actions"].astype(np.float32)
    masks = cache["token_mask"][:, -1]  # last future frame
    has_mask = bool(cache["has_mask"].all())
    scene_ids = [r["pair_id"] for r in rows]
    groups = group_cells(rows)
    pooled_ctx = context.reshape(len(context), -1, context.shape[-1]).mean(1)

    preds: dict[str, np.ndarray | None] = {}
    notes: dict[str, str] = {}
    preds["persistence"] = np.repeat(context[:, None], 1, axis=1)[:, 0]
    preds["loso_ridge"] = loso_ridge(context, target, actions, scene_ids, lam)
    cds, donors = copy_delta_scene(context, target, pooled_ctx, actions, scene_ids, action_weight)
    preds["copy_delta_scene"] = cds
    if "droid_pooled_t" in cache and len(cache["droid_pooled_t"]) > 0:
        preds["knn_droid"] = knn_droid(context, pooled_ctx, actions, cache["droid_pooled_t"].astype(np.float32), cache["droid_delta"], cache["droid_actions"].astype(np.float32), k, action_weight)
        notes["knn_droid"] = f"k={k} over {len(cache['droid_pooled_t'])} DROID transitions"
    else:
        preds["knn_droid"] = None
        notes["knn_droid"] = "deferred: cache has no DROID transition latents (build with latent_cache.py --droid-reference)"

    model_cos = per_scene_cosines(model_pred, target, groups, masks)
    result: dict[str, Any] = {
        "n_scenes": len(groups),
        "token_mask": f"{REGION_LABEL[domain]} patches from masks/" if has_mask else "all_tokens_no_masks",
        "model": {"per_scene_cosine": model_cos, "mean_cosine": cluster_bootstrap_mean(np.array(list(model_cos.values()))), "sign_flip_p_vs_zero": sign_flip_p(np.array(list(model_cos.values())))},
        "baselines": {},
        "notes": notes,
        "donor_map_copy_delta_scene": {rows[i]["cell_id"]: rows[int(d)]["cell_id"] for i, d in enumerate(donors)},
    }
    gate_parts = {}
    for name, y in preds.items():
        if y is None:
            result["baselines"][name] = {"status": "deferred"}
            gate_parts[name] = None
            continue
        bc = per_scene_cosines(y, target, groups, masks)
        cmp = compare_to_model(model_cos, bc)
        result["baselines"][name] = {"per_scene_cosine": bc, **cmp}
        gate_parts[name] = bool(cmp["diff"]["mean"] > 0 and cmp["sign_flip_p"] < 0.05)
    result["sonar"] = projected_scene_cosines({"model": model_pred, **preds}, target, groups, masks)
    if domain != "egg":
        groups3 = identity_cells(rows)
        result["domain"] = domain
        result["identity"] = identity_block({"model": model_pred, **preds}, target, groups3, masks) if groups3 else {"status": "absent", "n_scenes": 0}
    evaluated = {k: v for k, v in gate_parts.items() if v is not None}
    result["interaction_gate"] = {
        "per_baseline": gate_parts,
        "passed": bool(evaluated and all(evaluated.values())),
        "deferred_baselines": [k for k, v in gate_parts.items() if v is None],
        "criterion": "model mean cosine(I_pred, I_true) > every baseline, sign-flip p < 0.05 over scenes",
        "min_attainable_p": 2.0 / (2 ** len(groups)) if groups else None,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="latent_cache.py npz")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-label", default="JEPA-WM DROID")
    ap.add_argument("--action-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--domain", choices=list(DOMAINS), default="egg")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in (args.artifacts / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    cache = load_cache(args.cache)
    if list(cache["cell_id"]) != [r["cell_id"] for r in rows]:
        raise SystemExit("cache cell order does not match manifest")
    result = analyze(cache, rows, args.action_weight, args.k, args.ridge_lambda, args.domain)
    result.update(
        {
            "model_label": args.model_label,
            "cache_meta": cache["meta"],
            "manifest_sha256": sha256_file(args.artifacts / "manifest.jsonl"),
            "cache_sha256": sha256_file(args.cache),
            "interpretation_scope": (
                "Interaction-level retrieval control. A failed gate means the predicted hazard x action interaction "
                "is not distinguishable from retrieval / scene-blind action response; mechanism claims are not licensed."
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(canonical_json(finite(result))), indent=2) + "\n")
    brief = {
        "n_scenes": result["n_scenes"],
        "token_mask": result["token_mask"],
        "model_mean_cosine": result["model"]["mean_cosine"],
        "baselines": {k: (v.get("baseline_mean_cosine"), v.get("sign_flip_p")) for k, v in result["baselines"].items()},
        "interaction_gate": result["interaction_gate"],
        **({"identity_model_mean_cosine": {k: result["identity"].get(k, {}).get("model", {}).get("mean_cosine") for k in ("object", "identity")}} if "identity" in result else {}),
        "sonar_k4": {f: result["sonar"].get(f, {}).get("k4") for f in ("pooled", "per_token")} if result["sonar"].get("status") is None else result["sonar"],
    }
    print(json.dumps(finite(brief), indent=2))


if __name__ == "__main__":
    main()
