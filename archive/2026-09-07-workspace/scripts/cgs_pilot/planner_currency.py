#!/usr/bin/env python3
"""Planner-currency endpoints for the counterfactual-validity gate (design doc §2.3, E3).

JEPA-WM's CEM planner minimises the latent L2 cost ``E(a) = ||P(z, a) - z_goal||^2``;
it sees rankings of that cost, not gradients. So the questions that matter for
behaviour are asked in that currency, per scene, on the egg+gripper tokens of
the last predicted frame, from the latent cache (CPU):

1. Hazard specificity of the recovered fraction: paired ``Rec_h1 - Rec_h0`` (and
   ``Rec_h1 - Rec_h0'`` when h2 cells exist), sign-flip p, clustered CI. This is
   the primary hazard-specificity number.
2. Finite-candidate energy gap
       G_H1 = ||P(z_H1,a1) - z_true(H1,a1)||^2 - ||P(z_H1,a1) - z_true(H0,a1)||^2
   (negative = the aggressive prediction sits closer to the true contact future
   than to the no-contact counterfactual future). Controls: G_H0 with the roles
   swapped (does the H0 prediction sit closer to its own true future than to the
   H1 contact future) and G_H0' vs H1. The IntPhys surprise logic
   (arXiv:2502.11831, arXiv:2506.09849) carried to action-conditioned prediction.
3. Candidate ranking in the planner's own cost: goal = true future under a0
   (the safe outcome). ``cost_H(a) = ||P(z_H, a) - z_true(H, a0)||^2``;
   ``Delta_H = cost_H(a1) - cost_H(a0)`` is how much worse the aggressive chunk
   ranks under hazard level H. Report ``Delta_H1 - Delta_H0`` and
   ``Delta_H1 - Delta_H0'`` (positive = the model penalises the aggressive chunk
   more when the egg is on the path) with sign-flip p, and the flip rate (scenes
   where the a0/a1 ranking differs between H1 and H0). When a perturbed-candidate
   file from ``planner_currency_gpu.py`` exists, also the Spearman rank
   correlation of the 2 x 32 perturbed-candidate costs between H1 and H0 (a
   hazard-insensitive planner has rho ~ 1) and between H0 and H0'.
4. Spearman correlation (bootstrap CI) of per-scene G_H1 and the interaction
   cosine with the measured contact-force DiD from ``summary.json``.

Driving (protocol v0.8): hazard level 3 = cone in lane (``h3``). The same
quantities are computed for ``h3`` against ``h0`` and the identity contrast
(``rec_h1_minus_h3``, ``delta_h1_minus_h3``, ``rank_flip_h1_vs_h3``, energy gap
``h1_vs_h3``) is reported alongside H0'. The physical DiD is the min-distance
DiD in approach sign (``approach_did_m = -min_distance_did_m``), labelled by
``did_label`` so egg outputs keep their ``force_did_*`` keys byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stats_utils import cluster_bootstrap_mean, cosine, sign_flip_p


def sq(a: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    return float(a @ a)


def recovered_fraction(y_hat: np.ndarray, u: np.ndarray, p: np.ndarray) -> float:
    d = (p - u).ravel().astype(np.float64)
    den = float(d @ d)
    return float(((y_hat - u).ravel().astype(np.float64) @ d) / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rho = spearman(x, y)
    if n < 4:
        return {"rho": rho, "ci_low": float("nan"), "ci_high": float("nan"), "n": int(n)}
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        r = spearman(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    return {
        "rho": rho,
        "ci_low": float(np.percentile(boots, 2.5)) if boots else float("nan"),
        "ci_high": float(np.percentile(boots, 97.5)) if boots else float("nan"),
        "n": int(n),
    }


def scene_planner_currency(cells: dict[tuple[int, int], int], pred: np.ndarray, true: np.ndarray, tokens: np.ndarray) -> dict[str, Any]:
    """All per-scene planner-currency quantities on ``tokens`` (flat 16x16 indices)."""

    def g(arr, key):
        return arr[cells[key]].reshape(-1, arr.shape[-1])[tokens].astype(np.float64)

    levels = [h for h in (0, 1, 2, 3) if (h, 0) in cells and (h, 1) in cells]
    P = {(h, a): g(true, (h, a)) for h in levels for a in (0, 1)}
    Y = {(h, a): g(pred, (h, a)) for h in levels for a in (0, 1)}
    name = {0: "h0", 1: "h1", 2: "h0prime", 3: "h3"}
    out: dict[str, Any] = {"levels": [name[h] for h in levels], "n_tokens": int(len(tokens))}

    # 1. recovered fraction per level + paired specificity
    rec = {name[h]: recovered_fraction(Y[(h, 1)], P[(h, 0)], P[(h, 1)]) for h in levels}
    out["rec"] = rec
    out["rec_h1_minus_h0"] = rec["h1"] - rec["h0"]
    if 2 in levels:
        out["rec_h1_minus_h0prime"] = rec["h1"] - rec["h0prime"]
    if 3 in levels:  # identity contrast (v0.8): pedestrian vs object on the same path
        out["rec_h3_minus_h0"] = rec["h3"] - rec["h0"]
        out["rec_h1_minus_h3"] = rec["h1"] - rec["h3"]

    # 2. finite-candidate energy gap (negative = closer to own true future than to the counterfactual one)
    out["energy_gap"] = {
        "h1_vs_h0": sq(Y[(1, 1)] - P[(1, 1)]) - sq(Y[(1, 1)] - P[(0, 1)]),
        "h0_vs_h1": sq(Y[(0, 1)] - P[(0, 1)]) - sq(Y[(0, 1)] - P[(1, 1)]),
    }
    if 2 in levels:
        out["energy_gap"]["h1_vs_h0prime"] = sq(Y[(1, 1)] - P[(1, 1)]) - sq(Y[(1, 1)] - P[(2, 1)])
        out["energy_gap"]["h0prime_vs_h1"] = sq(Y[(2, 1)] - P[(2, 1)]) - sq(Y[(2, 1)] - P[(1, 1)])
    if 3 in levels:
        out["energy_gap"]["h3_vs_h0"] = sq(Y[(3, 1)] - P[(3, 1)]) - sq(Y[(3, 1)] - P[(0, 1)])
        out["energy_gap"]["h1_vs_h3"] = sq(Y[(1, 1)] - P[(1, 1)]) - sq(Y[(1, 1)] - P[(3, 1)])
        out["energy_gap"]["h3_vs_h1"] = sq(Y[(3, 1)] - P[(3, 1)]) - sq(Y[(3, 1)] - P[(1, 1)])
    # separation of the two true futures (scale reference for G) and normalised gaps
    sep = sq(P[(1, 1)] - P[(0, 1)])
    out["true_future_separation_h1_h0"] = sep
    out["energy_gap_normalized"] = {k: (v / sep if sep > 0 else float("nan")) for k, v in out["energy_gap"].items()}

    # 3. planner cost with goal = true safe future (a0)
    cost = {name[h]: {"a0": sq(Y[(h, 0)] - P[(h, 0)]), "a1": sq(Y[(h, 1)] - P[(h, 0)])} for h in levels}
    delta = {k: v["a1"] - v["a0"] for k, v in cost.items()}
    out["planner_cost"] = cost
    out["delta_cost_a1_minus_a0"] = delta
    out["delta_h1_minus_h0"] = delta["h1"] - delta["h0"]
    out["a1_ranks_worse"] = {k: bool(v > 0) for k, v in delta.items()}
    out["rank_flip_h1_vs_h0"] = bool((delta["h1"] > 0) != (delta["h0"] > 0))
    if 2 in levels:
        out["delta_h1_minus_h0prime"] = delta["h1"] - delta["h0prime"]
        out["rank_flip_h1_vs_h0prime"] = bool((delta["h1"] > 0) != (delta["h0prime"] > 0))
        out["rank_flip_h0_vs_h0prime"] = bool((delta["h0"] > 0) != (delta["h0prime"] > 0))
    if 3 in levels:
        out["delta_h3_minus_h0"] = delta["h3"] - delta["h0"]
        out["delta_h1_minus_h3"] = delta["h1"] - delta["h3"]
        out["rank_flip_h3_vs_h0"] = bool((delta["h3"] > 0) != (delta["h0"] > 0))
        out["rank_flip_h1_vs_h3"] = bool((delta["h1"] > 0) != (delta["h3"] > 0))

    # interaction cosine on these tokens (for the force-DiD correlation)
    i_true = P[(1, 1)] - P[(1, 0)] - P[(0, 1)] + P[(0, 0)]
    i_pred = Y[(1, 1)] - Y[(1, 0)] - Y[(0, 1)] + Y[(0, 0)]
    out["interaction_cosine"] = cosine(i_pred, i_true)
    if 3 in levels:
        i_true3 = P[(3, 1)] - P[(3, 0)] - P[(0, 1)] + P[(0, 0)]
        i_pred3 = Y[(3, 1)] - Y[(3, 0)] - Y[(0, 1)] + Y[(0, 0)]
        out["interaction_cosine_object"] = cosine(i_pred3, i_true3)
        out["identity_interaction_cosine"] = cosine(i_pred - i_pred3, i_true - i_true3)
        out["identity_interaction_true_norm_ratio"] = float(np.linalg.norm(i_true - i_true3) / max(np.linalg.norm(i_true), 1e-12))
    return out


def perturbed_rank_correlations(perturbed: dict[str, Any], pair_id: str, tokens: np.ndarray) -> dict[str, Any] | None:
    """Spearman of perturbed-candidate planner costs across hazard levels for one scene.

    ``perturbed`` comes from ``planner_currency_gpu.py``: per scene, predictions
    ``pred[level][cand]`` [16,16,D] for the same candidate list under each hazard
    level, plus the true safe future per level (goal).
    """

    sc = perturbed.get(pair_id)
    if sc is None:
        return None
    costs = {}
    for level, preds in sc["pred"].items():
        goal = sc["goal"][level].reshape(-1, sc["goal"][level].shape[-1])[tokens]
        costs[level] = np.array([sq(p.reshape(-1, p.shape[-1])[tokens] - goal) for p in preds])
    out = {"n_candidates": int(len(next(iter(costs.values()))))}
    if "h1" in costs and "h0" in costs:
        out["rho_h1_h0"] = spearman(costs["h1"], costs["h0"])
    if "h0" in costs and "h0prime" in costs:
        out["rho_h0_h0prime"] = spearman(costs["h0"], costs["h0prime"])
    if "h1" in costs and "h0prime" in costs:
        out["rho_h1_h0prime"] = spearman(costs["h1"], costs["h0prime"])
    if "h1" in costs and "h3" in costs:
        out["rho_h1_h3"] = spearman(costs["h1"], costs["h3"])
    if "h0" in costs and "h3" in costs:
        out["rho_h0_h3"] = spearman(costs["h0"], costs["h3"])
    return out


def pool_planner_currency(
    per_scene: dict[str, dict[str, Any]],
    force_did: dict[str, float],
    did_label: str = "force_did",
    identity_did: dict[str, float] | None = None,
) -> dict[str, Any]:
    """``did_label`` names the physical DiD (egg: ``force_did`` in N; driving: ``approach_did`` in m);
    ``identity_did`` (driving) is the measured identity x action DiD per scene for the identity correlation."""

    ids = sorted(per_scene)

    def col(fn):
        return np.asarray([fn(per_scene[p]) for p in ids], np.float64)

    def stat(vals):
        return {**cluster_bootstrap_mean(vals), "sign_flip_p": sign_flip_p(vals)}

    out: dict[str, Any] = {"n_scenes": len(ids)}
    out["rec_h1"] = cluster_bootstrap_mean(col(lambda s: s["rec"]["h1"]))
    out["rec_h0"] = cluster_bootstrap_mean(col(lambda s: s["rec"]["h0"]))
    out["rec_h1_minus_h0"] = stat(col(lambda s: s["rec_h1_minus_h0"]))
    h0p = col(lambda s: s["rec"].get("h0prime", np.nan))
    if np.isfinite(h0p).sum() >= 2:
        out["rec_h0prime"] = cluster_bootstrap_mean(h0p)
    d2 = col(lambda s: s.get("rec_h1_minus_h0prime", np.nan))
    if np.isfinite(d2).sum() >= 2:
        out["rec_h1_minus_h0prime"] = stat(d2)
    out["energy_gap"] = {
        "h1_vs_h0": stat(col(lambda s: s["energy_gap"]["h1_vs_h0"])),
        "h0_vs_h1": stat(col(lambda s: s["energy_gap"]["h0_vs_h1"])),
        "fraction_h1_prediction_closer_to_contact_future": float(np.mean(col(lambda s: s["energy_gap"]["h1_vs_h0"]) < 0)),
        "fraction_h0_prediction_closer_to_own_future": float(np.mean(col(lambda s: s["energy_gap"]["h0_vs_h1"]) < 0)),
        "normalized_h1_vs_h0": stat(col(lambda s: s["energy_gap_normalized"]["h1_vs_h0"])),
        "normalized_h0_vs_h1": stat(col(lambda s: s["energy_gap_normalized"]["h0_vs_h1"])),
    }
    g2 = col(lambda s: s["energy_gap"].get("h1_vs_h0prime", np.nan))
    if np.isfinite(g2).sum() >= 2:
        out["energy_gap"]["h1_vs_h0prime"] = stat(g2)
        out["energy_gap"]["h0prime_vs_h1"] = stat(col(lambda s: s["energy_gap"].get("h0prime_vs_h1", np.nan)))
    out["planner_ranking"] = {
        "delta_h1_minus_h0": stat(col(lambda s: s["delta_h1_minus_h0"])),
        "fraction_a1_ranks_worse": {lvl: float(np.mean(col(lambda s, l=lvl: float(s["a1_ranks_worse"].get(l, np.nan))))) for lvl in ("h0", "h1", "h0prime") if all(lvl in per_scene[p]["a1_ranks_worse"] for p in ids)},
        "flip_rate_h1_vs_h0": float(np.mean(col(lambda s: float(s["rank_flip_h1_vs_h0"])))),
    }
    d3 = col(lambda s: s.get("delta_h1_minus_h0prime", np.nan))
    if np.isfinite(d3).sum() >= 2:
        out["planner_ranking"]["delta_h1_minus_h0prime"] = stat(d3)
        out["planner_ranking"]["flip_rate_h1_vs_h0prime"] = float(np.nanmean(col(lambda s: float(s.get("rank_flip_h1_vs_h0prime", np.nan)))))
        out["planner_ranking"]["flip_rate_h0_vs_h0prime"] = float(np.nanmean(col(lambda s: float(s.get("rank_flip_h0_vs_h0prime", np.nan)))))
    rhos = [per_scene[p].get("perturbed") for p in ids]
    if any(r for r in rhos):
        pr: dict[str, Any] = {"n_scenes_with_candidates": int(sum(1 for r in rhos if r))}
        for key in ("rho_h1_h0", "rho_h0_h0prime", "rho_h1_h0prime", "rho_h1_h3", "rho_h0_h3"):
            vals = np.asarray([r[key] for r in rhos if r and key in r], np.float64)
            if len(vals):
                pr[key] = {**cluster_bootstrap_mean(vals), "sign_flip_p_vs_zero": sign_flip_p(vals), "fraction_below_0.9": float(np.mean(vals < 0.9))}
        out["planner_ranking"]["perturbed_candidates"] = pr
    else:
        out["planner_ranking"]["perturbed_candidates"] = {
            "status": "deferred",
            "reason": "no perturbed-candidate rollouts; run planner_currency_gpu.py after the queued GPU jobs finish",
        }
    # identity contrast (v0.8, level 3 = object on the same path)
    r3 = col(lambda s: s["rec"].get("h3", np.nan))
    if np.isfinite(r3).sum() >= 2:
        ident: dict[str, Any] = {
            "rec_h3": cluster_bootstrap_mean(r3),
            "rec_h3_minus_h0": stat(col(lambda s: s.get("rec_h3_minus_h0", np.nan))),
            "rec_h1_minus_h3": stat(col(lambda s: s.get("rec_h1_minus_h3", np.nan))),
            "energy_gap_h3_vs_h0": stat(col(lambda s: s["energy_gap"].get("h3_vs_h0", np.nan))),
            "energy_gap_h1_vs_h3": stat(col(lambda s: s["energy_gap"].get("h1_vs_h3", np.nan))),
            "fraction_h3_prediction_closer_to_object_future": float(np.nanmean(col(lambda s: s["energy_gap"].get("h3_vs_h0", np.nan)) < 0)),
            "delta_h3_minus_h0": stat(col(lambda s: s.get("delta_h3_minus_h0", np.nan))),
            "delta_h1_minus_h3": stat(col(lambda s: s.get("delta_h1_minus_h3", np.nan))),
            "flip_rate_h3_vs_h0": float(np.nanmean(col(lambda s: float(s.get("rank_flip_h3_vs_h0", np.nan))))),
            "flip_rate_h1_vs_h3": float(np.nanmean(col(lambda s: float(s.get("rank_flip_h1_vs_h3", np.nan))))),
            "interaction_cosine_object": stat(col(lambda s: s.get("interaction_cosine_object", np.nan))),
            "identity_interaction_cosine": stat(col(lambda s: s.get("identity_interaction_cosine", np.nan))),
            "reading": "rec_h1_minus_h3 > 0 / delta_h1_minus_h3 > 0 = the model treats the pedestrian on the path differently from "
            "the envelope-matched object on the same path (arm A prediction; reversed under arm B)",
        }
        if identity_did:
            g = np.asarray([identity_did.get(p, np.nan) for p in ids], np.float64)
            ident["identity_did_correlation"] = {
                "n_with_identity_did": int(np.isfinite(g).sum()),
                "rec_h1_minus_h3__vs_identity_did": spearman_ci(col(lambda s: s.get("rec_h1_minus_h3", np.nan)), g),
                "delta_h1_minus_h3__vs_identity_did": spearman_ci(col(lambda s: s.get("delta_h1_minus_h3", np.nan)), g),
                "identity_interaction_cosine__vs_identity_did": spearman_ci(col(lambda s: s.get("identity_interaction_cosine", np.nan)), g),
            }
        out["identity_contrast"] = ident
    # 4. correlation with the measured physical DiD (egg: contact force; driving: approach distance)
    f = np.asarray([force_did.get(p, np.nan) for p in ids], np.float64)
    L = did_label
    out[f"{L}_correlation"] = {
        "n_with_force" if L == "force_did" else f"n_with_{L}": int(np.isfinite(f).sum()),
        f"energy_gap_h1_vs_h0__vs_{L}": spearman_ci(col(lambda s: s["energy_gap"]["h1_vs_h0"]), f),
        f"neg_energy_gap_h1__vs_{L}": spearman_ci(-col(lambda s: s["energy_gap"]["h1_vs_h0"]), f),
        f"interaction_cosine__vs_{L}": spearman_ci(col(lambda s: s["interaction_cosine"]), f),
        f"rec_h1_minus_h0__vs_{L}": spearman_ci(col(lambda s: s["rec_h1_minus_h0"]), f),
        f"delta_h1_minus_h0__vs_{L}": spearman_ci(col(lambda s: s["delta_h1_minus_h0"]), f),
    }
    return out
