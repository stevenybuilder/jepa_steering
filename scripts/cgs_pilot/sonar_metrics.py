#!/usr/bin/env python3
"""Model-native contrast-subspace ("sonar") versions of the counterfactual metrics.

Motivation. The raw metrics compare predicted and true hazard x action
interaction vectors in the full 1024-d per-token DINOv3 coordinate system with
plain Euclidean geometry. A low raw cosine can mean the model lacks the
representation, or that the shared, hazard-specific contrast lives in a
low-dimensional subspace swamped by high-dimensional, scene-specific variance.
The sonar metrics ask the second question directly: build, from OTHER scenes'
TRUE-future interaction vectors, the direction / top-k subspace in which the
true contrast is shared, and score the held-out scene inside that subspace.

Everything is leave-one-scene-out (LOSO): the subspace used to score scene i is
fit on scenes j != i only, so no statistic is computed on data that shaped its
own coordinate system (cross-validated statistics in the sense of
arXiv:2510.00845; Hastie, Tibshirani & Friedman ch. 7 on the optimism of
in-sample estimates). Subspaces are UNCENTERED singular subspaces (top-k right
singular vectors of the stacked vectors) so the shared mean direction is
representable at k = 1.

Two feature spaces:

- ``pooled``:    one D-vector per scene = mean over the group's tokens. n-1
                 training vectors per fold; k is capped at n-2.
- ``per_token``: every token of every other scene is a training vector in R^D;
                 the held-out scene's tokens are projected individually and the
                 concatenation is scored. Aligns across scenes with different
                 token sets.

Per scene i, group g, feature space f and k:

- projected cosine / NMSE between P I_pred_i and P I_true_i;
- capture = ||P I_true_i||^2 / ||I_true_i||^2 (how much of the true contrast the
  shared subspace explains);
- projected Rec for H1 (U = true (H1,A0), P = true (H1,A1), Y = pred (H1,A1)),
  H0 control, and H0' (h2 cells) when present, with all three vectors projected;
- controls: a random orthonormal k-subspace (expect cosine ~ 0, capture ~ k/D),
  the action-main-effect subspace (built from other scenes' H-averaged
  A1 - A0 true-future vectors), a scene-shuffled subspace (built from
  contrasts with the quartet labels permuted within each other scene), and the
  principal angles between the interaction and action-main-effect subspaces.

Hazard specificity: paired Rec_h1 - Rec_h0 per scene (raw and projected) with
a sign-flip test; this is the number that says whether the model treats the
on-path egg differently from the off-path egg at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stats_utils import cluster_bootstrap_mean, cosine, nmse, sign_flip_p

KS = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Linear algebra helpers
# --------------------------------------------------------------------------- #


def svd_subspace(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k uncentered right singular vectors of X [n, D] -> Q [D, k] orthonormal.

    Uses the D x D Gram matrix when n > D (many tokens, D = 1024), which is much
    cheaper than a thin SVD of X and yields the same subspace.
    """

    X = np.asarray(X, np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if X.size == 0:
        raise ValueError("no training vectors")
    n, D = X.shape
    k = max(1, min(k, min(n, D)))
    if n > D:
        G = X.T @ X
        try:
            from scipy.linalg import eigh as sp_eigh

            _, v = sp_eigh(G, subset_by_index=[D - k, D - 1])
            return v[:, ::-1]
        except Exception:  # pragma: no cover - scipy missing
            _, v = np.linalg.eigh(G)
            return v[:, ::-1][:, :k]
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:k].T


def unit_mean_direction(X: np.ndarray) -> np.ndarray:
    m = np.asarray(X, np.float64).mean(axis=0)
    n = np.linalg.norm(m)
    return (m / n)[:, None] if n > 0 else np.zeros((len(m), 1))


def random_subspace(D: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    q, _ = np.linalg.qr(rng.randn(D, k))
    return q


def project(V: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project rows of V [..., D] onto span(Q) (returned in ambient coordinates)."""

    V = np.asarray(V, np.float64)
    shape = V.shape
    flat = V.reshape(-1, shape[-1])
    return ((flat @ Q) @ Q.T).reshape(shape)


def principal_angles_deg(Q1: np.ndarray, Q2: np.ndarray) -> list[float]:
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s)).tolist()


def recovered_fraction(y_hat: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
    d = (p - u).ravel()
    den = float(d @ d)
    return float(((y_hat - u).ravel() @ d) / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Scene-level containers
# --------------------------------------------------------------------------- #


def scene_contrasts(cells: dict[tuple[int, int], int], pred: np.ndarray, true: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
    """Per-scene contrast vectors restricted to ``tokens`` (flat indices into the 16x16 grid).

    ``pred``/``true`` are [n_cells, 16, 16, D]. Returns [n_tok, D] arrays.
    """

    def g(arr, key):
        return arr[cells[key]].reshape(-1, arr.shape[-1])[tokens].astype(np.float64)

    P = {k: g(true, k) for k in cells}
    Y = {k: g(pred, k) for k in cells}
    out = {
        "tokens": tokens,
        "i_true": P[(1, 1)] - P[(1, 0)] - P[(0, 1)] + P[(0, 0)],
        "i_pred": Y[(1, 1)] - Y[(1, 0)] - Y[(0, 1)] + Y[(0, 0)],
        "a_true": 0.5 * ((P[(1, 1)] - P[(1, 0)]) + (P[(0, 1)] - P[(0, 0)])),
        "rec": {
            "h1": (Y[(1, 1)], P[(1, 0)], P[(1, 1)]),
            "h0": (Y[(0, 1)], P[(0, 0)], P[(0, 1)]),
        },
        "quartet_true": {k: P[k] for k in ((0, 0), (0, 1), (1, 0), (1, 1))},
    }
    if (2, 0) in cells and (2, 1) in cells:
        out["rec"]["h0prime"] = (Y[(2, 1)], P[(2, 0)], P[(2, 1)])
    return out


def shuffled_contrast(quartet: dict[tuple[int, int], np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    perm = rng.permutation(4)
    q = {keys[i]: quartet[keys[perm[i]]] for i in range(4)}
    return q[(1, 1)] - q[(1, 0)] - q[(0, 1)] + q[(0, 0)]


def _pool(v: np.ndarray) -> np.ndarray:
    return v.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# LOSO evaluation
# --------------------------------------------------------------------------- #


def _score(sc: dict[str, Any], Q: np.ndarray, feature: str) -> dict[str, Any]:
    """Score one held-out scene inside span(Q)."""

    def rep(v):  # representation used for scoring
        return _pool(v) if feature == "pooled" else v

    it, ip = rep(sc["i_true"]), rep(sc["i_pred"])
    pit, pip = project(it, Q), project(ip, Q)
    den = float(np.sum(it**2))
    out = {
        "cosine": cosine(pip, pit),
        "nmse": nmse(pip, pit),
        "capture": float(np.sum(pit**2) / den) if den > 0 else float("nan"),
        "raw_cosine": cosine(ip, it),
        "rec": {},
    }
    for name, (y, u, p) in sc["rec"].items():
        out["rec"][name] = recovered_fraction(project(rep(y), Q), project(rep(u), Q), project(rep(p), Q))
        out["rec"][f"{name}_raw"] = recovered_fraction(rep(y), rep(u), rep(p))
    return out


def _train_matrix(scenes: list[dict[str, Any]], key: str, feature: str) -> np.ndarray:
    vecs = [(_pool(s[key]) if feature == "pooled" else s[key]) for s in scenes]
    return np.concatenate(vecs, axis=0)


def loso_sonar(scenes: dict[str, dict[str, Any]], ks: tuple[int, ...] = KS, seed: int = 0) -> dict[str, Any]:
    """scenes: pair_id -> scene_contrasts(...) for one token group."""

    ids = sorted(scenes)
    n = len(ids)
    if n < 3:
        return {"status": "insufficient_scenes", "n_scenes": n}
    D = scenes[ids[0]]["i_true"].shape[-1]
    rng = np.random.RandomState(seed)
    per: dict[str, dict[str, Any]] = {}
    for feature in ("pooled", "per_token"):
        per[feature] = {}
        for i, pid in enumerate(ids):
            others = [scenes[o] for o in ids if o != pid]
            X_int = _train_matrix(others, "i_true", feature)
            X_act = _train_matrix(others, "a_true", feature)
            X_shuf = np.concatenate(
                [(_pool(v) if feature == "pooled" else v) for v in (shuffled_contrast(o["quartet_true"], rng) for o in others)], axis=0
            )
            res: dict[str, Any] = {"mean_direction": _score(scenes[pid], unit_mean_direction(X_int), feature)}
            kmax = max(ks)
            cap = (len(X_int) - 1) if feature == "pooled" else kmax
            # Top-k singular subspaces are nested: one decomposition per fold, sliced per k.
            Qi_full = svd_subspace(X_int, min(kmax, cap))
            Qa_full = svd_subspace(X_act, min(kmax, cap))
            Qs_full = svd_subspace(X_shuf, min(kmax, cap))
            Qr_full = random_subspace(D, min(kmax, cap), rng)
            for k in ks:
                kk = min(k, Qi_full.shape[1])
                Qi, Qa, Qs, Qr = Qi_full[:, :kk], Qa_full[:, :kk], Qs_full[:, :kk], Qr_full[:, :kk]
                res[f"k{k}"] = {
                    "k_used": int(kk),
                    "interaction": _score(scenes[pid], Qi, feature),
                    "random": _score(scenes[pid], Qr, feature),
                    "action_main_effect": _score(scenes[pid], Qa, feature),
                    "shuffled_scene": _score(scenes[pid], Qs, feature),
                    "principal_angles_interaction_vs_action_deg": principal_angles_deg(Qi, Qa),
                }
            per[feature][pid] = res
    return {"status": "ok", "n_scenes": n, "per_scene": per, "pooled": _pool_scenes(per, ids, ks)}


def _collect(per_feature: dict[str, Any], ids: list[str], path: list[str]) -> np.ndarray:
    """Gather one scalar per scene along ``path``; missing keys (e.g. h0prime) -> nan."""

    vals = []
    for pid in ids:
        node = per_feature[pid]
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                break
        vals.append(float("nan") if node is None else node)
    return np.asarray(vals, np.float64)


def _pool_scenes(per: dict[str, Any], ids: list[str], ks: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature, pf in per.items():
        f: dict[str, Any] = {}
        raw = _collect(pf, ids, ["mean_direction", "raw_cosine"])
        f["raw_cosine"] = {**cluster_bootstrap_mean(raw), "sign_flip_p": sign_flip_p(raw)}
        md = _collect(pf, ids, ["mean_direction", "cosine"])
        f["mean_direction"] = {
            "cosine": {**cluster_bootstrap_mean(md), "sign_flip_p": sign_flip_p(md)},
            "capture": cluster_bootstrap_mean(_collect(pf, ids, ["mean_direction", "capture"])),
        }
        for k in ks:
            kb: dict[str, Any] = {}
            for sub in ("interaction", "random", "action_main_effect", "shuffled_scene"):
                c = _collect(pf, ids, [f"k{k}", sub, "cosine"])
                kb[sub] = {
                    "cosine": {**cluster_bootstrap_mean(c), "sign_flip_p": sign_flip_p(c)},
                    "nmse": cluster_bootstrap_mean(_collect(pf, ids, [f"k{k}", sub, "nmse"])),
                    "capture": cluster_bootstrap_mean(_collect(pf, ids, [f"k{k}", sub, "capture"])),
                }
                if sub == "interaction":
                    rec_names = sorted({rn for pid in ids for rn in pf[pid][f"k{k}"][sub]["rec"]})
                    kb[sub]["rec"] = {rn: cluster_bootstrap_mean(_collect(pf, ids, [f"k{k}", sub, "rec", rn])) for rn in rec_names}
                    for a, b, lab in (("h1", "h0", "h1_minus_h0"), ("h1", "h0prime", "h1_minus_h0prime")):
                        if a in rec_names and b in rec_names:
                            d = _collect(pf, ids, [f"k{k}", sub, "rec", a]) - _collect(pf, ids, [f"k{k}", sub, "rec", b])
                            kb[sub]["rec"][lab] = {**cluster_bootstrap_mean(d), "sign_flip_p": sign_flip_p(d)}
                            d_raw = _collect(pf, ids, [f"k{k}", sub, "rec", f"{a}_raw"]) - _collect(pf, ids, [f"k{k}", sub, "rec", f"{b}_raw"])
                            kb[sub]["rec"][f"{lab}_raw"] = {**cluster_bootstrap_mean(d_raw), "sign_flip_p": sign_flip_p(d_raw)}
            diff = _collect(pf, ids, [f"k{k}", "interaction", "cosine"]) - _collect(pf, ids, [f"k{k}", "random", "cosine"])
            kb["interaction_minus_random_cosine"] = {**cluster_bootstrap_mean(diff), "sign_flip_p": sign_flip_p(diff)}
            diff_a = _collect(pf, ids, [f"k{k}", "interaction", "cosine"]) - _collect(pf, ids, [f"k{k}", "action_main_effect", "cosine"])
            kb["interaction_minus_action_cosine"] = {**cluster_bootstrap_mean(diff_a), "sign_flip_p": sign_flip_p(diff_a)}
            # scenes can have fewer than k principal angles (group with < k tokens): NaN-pad to a rectangular array
            ang_lists = [list(pf[pid][f"k{k}"]["principal_angles_interaction_vs_action_deg"]) for pid in ids]
            width = max((len(a) for a in ang_lists), default=0)
            angles = np.full((len(ang_lists), max(width, 1)), np.nan, np.float64)
            for i, a in enumerate(ang_lists):
                angles[i, : len(a)] = a
            with np.errstate(all="ignore"):
                kb["principal_angles_interaction_vs_action_deg"] = {
                    "min_mean": float(np.nanmean(np.nanmin(angles, axis=1))),
                    "max_mean": float(np.nanmean(np.nanmax(angles, axis=1))),
                    "per_scene_min": np.nanmin(angles, axis=1).tolist(),
                    "n_scenes_short_angles": int(sum(len(a) < width for a in ang_lists)),
                }
            f[f"k{k}"] = kb
        out[feature] = f
    return out


def interpretation(sonar: dict[str, Any], raw_cos: float, groups: tuple[str, ...] = ("egg", "gripper_corridor"), k: int = 4) -> str:
    gs = sonar.get("groups", {})
    if not any(gs.get(g, {}).get("status") == "ok" for g in groups):
        groups = tuple(g for g, v in gs.items() if v.get("status") == "ok")[:2]
    parts = []
    for group in groups:
        g = gs.get(group)
        if not g or g.get("status") != "ok":
            continue
        f = g["pooled"]["per_token"]
        kb = f.get(f"k{k}") or f[f"k{max(KS)}"]
        it, rd, ac, sh = kb["interaction"], kb["random"], kb["action_main_effect"], kb["shuffled_scene"]
        rec = it["rec"]
        spec, spec_raw = rec.get("h1_minus_h0", {}), rec.get("h1_minus_h0_raw", {})
        h0p = rec.get("h0prime")
        parts.append(
            f"[{group}, per-token, k={k}] raw cosine {f['raw_cosine']['mean']:.2f} -> projected {it['cosine']['mean']:.2f} "
            f"(sign-flip p {it['cosine']['sign_flip_p']:.3g}); controls: random subspace {rd['cosine']['mean']:.2f}, "
            f"action-main-effect subspace {ac['cosine']['mean']:.2f}, quartet-shuffled subspace {sh['cosine']['mean']:.2f}; "
            f"capture {100 * it['capture']['mean']:.0f}% (random {100 * rd['capture']['mean']:.1f}%); smallest principal angle "
            f"to the action subspace {kb['principal_angles_interaction_vs_action_deg']['min_mean']:.0f} deg. Projected Rec "
            f"H1 {rec['h1']['median']:.2f} vs H0 {rec['h0']['median']:.2f}"
            + (f" vs H0' {h0p['median']:.2f}" if h0p and h0p.get("n") else "")
            + f"; specificity Rec_h1 - Rec_h0 = {spec.get('mean', float('nan')):+.3f} (p {spec.get('sign_flip_p', float('nan')):.2g}), "
            f"raw-space specificity {spec_raw.get('mean', float('nan')):+.3f} (p {spec_raw.get('sign_flip_p', float('nan')):.2g})."
        )
    if not parts:
        return "Sonar metrics unavailable (fewer than 3 complete scenes with tokens in the requested groups)."
    return (
        "Model-native contrast subspace, leave-one-scene-out (the subspace scoring scene i is fit on the other scenes' TRUE "
        "interactions only; cross-validated statistics in the sense of arXiv:2510.00845). "
        + " ".join(parts)
        + " Reading guide: a random k-subspace reproduces the raw cosine (it neither helps nor hurts), so the gain over the "
        "random control is the coordinate-system effect; if the quartet-shuffled subspace scores as well as the interaction "
        "subspace, the shared directions are generic scene/action structure rather than an interaction-specific code. "
        "Pooled-feature Rec at small k has tiny denominators and is unstable; use per-token medians. k=1 projected cosine is "
        "a sign test (+-1 per scene). The specificity p (Rec_h1 - Rec_h0) is the number that says whether the model treats "
        "the on-path egg differently from the off-path egg; a large p means it does not, in raw or shared coordinates."
    )
