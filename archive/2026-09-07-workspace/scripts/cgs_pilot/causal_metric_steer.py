#!/usr/bin/env python3
"""Operator 1 -- minimum-distortion steering edit under the causal (nuisance-whitened) metric.

Design-doc references: ``cross model design jepa.md`` Step 4 ("nuisance-whitened metric <x,y>_N = x^T (Sigma_N + lambda I)^-1 y
estimated on fitting scenes from independently varied non-relational factors") and Step 8 ("minimum-distortion constraint:
maximize Delta relational energy subject to ||delta||_N <= eps, D_action <= tau, D_local <= tau_d");
``representational_geometry_paper_concepts.md`` section 1 (causal inner product / whitening / intervention representation:
"adding a vector changes the concept value while leaving causally separable concepts unchanged"; the metric is part of the
representation hypothesis), section 6 (conceptor = covariance-weighted soft subspace; matched-spectrum random control),
section 12.1 (Mahalanobis ellipsoid, whitening maps it to a sphere; C^-1 penalises low-variance axes), section 12.2 (a
structured method must beat a matched Gaussian / random baseline), section 12.5 (pointwise vs subspace intervention) and
section 14 ("variance is not semantics", "a global vector can hide structured residuals").

WHAT IS FITTED (per site x imagined step x token group, fitting = discovery scenes only)
--------------------------------------------------------------------------------------
1. Nuisance second moment  Sigma_N = sum_g w_g (1/n_g) sum_i n_{g,i} n_{g,i}^T + lambda I,  from displacement rows produced by
   varying NON-safety factors only (no in-lane solid identity anywhere):
     - ``scene``   hazard-free cells (levels 0 = sidewalk pedestrian, 2 = H0' mirror pose) centred within (level, action):
                   scene identity, hazard pixel position, and -- on v0.9 stimuli -- distance, prefix throttle and lateral offset;
     - ``action``  hazard-free action main effect  h[l,1] - h[l,0]  (l in {0, 2}), uncentred;
     - ``null``    H0' displacement  h[2,a] - h[0,a]  (mirror pose = position/appearance nuisance), uncentred;
     - ``factors`` ridge-regression displacement per 1 SD of each randomised v0.9 factor (hazard_dist_m, prefix_throttle,
                   hazard_lateral_offset_m) when the manifest carries them with non-degenerate range;
     - ``token``   within-cell token deviations of the hazard-free cells (the edit is written per token, so token-position
                   variation is nuisance too).
   lambda = lam_rel * tr(Sigma_N)/d (ridge floor for directions no nuisance factor ever varied; n_rows << d).
   Metric M = (Sigma_N)^-1, ||delta||_N^2 = delta^T M delta  (Mahalanobis in nuisance units, section 12.1).
2. Relational target R = (1/S) sum_i r_i r_i^T, r_i = DiD_solid,i - DiD_null,i projected off the rank-k hazard-free action
   subspace Q_act (exactly ``geometry_cross_arm.relational_rows``; the transport export's rel_* conceptors are the
   aperture-softened version of the same rows, section 6).  Relational energy in the CAUSAL inner product:
       E_N(delta) = (1/S) sum_i <delta, r_i>_N^2 = delta^T (M R M) delta.
   (Euclidean energy delta^T R delta is the twin with M = I / sigma_bar^2, sigma_bar^2 = tr(Sigma_N)/d.)
3. Constraints written as quadratic forms in delta:
     D_action(delta) = delta^T Q_act Q_act^T delta  <= tau^2,  tau = tau_action_rel * median_i ||Q_act^T a_i||  (a_i hazard-free action rows);
     D_local: diagonal receiver Mahalanobis of ``patch_site.mahalanobis_score`` (sqrt(mean_d z^2) ~ 1 in distribution);
              sqrt(delta^T S_r delta) <= mahal_incr with S_r = diag(1/var_r)/d bounds the increment by the triangle inequality;
     optional pullback G = J^T W J from the existing forward-AD JVP (``action_jacobian_sonar.downstream_response``): the
              ``pullback`` sub-command measures ||J_hf w_j|| of every operator direction relative to ||J_hf a|| (the JVP of
              the hazard-free action row) and writes a dose cap eps_cap; it needs the model and so runs on the box.

CLOSED FORM (Lagrangian / generalised eigenproblem) -- stated as required
------------------------------------------------------------------------
maximise  delta^T A delta,  A = M R M,   s.t.  delta^T M delta <= eps^2,  delta^T B delta <= tau^2,  delta^T S_r delta <= tau_d^2.
Lagrangian  L = delta^T A delta - alpha (delta^T M delta - eps^2) - beta (delta^T B delta - tau^2) - gamma (delta^T S_r delta - tau_d^2);
stationarity  A delta = (alpha M + beta B + gamma S_r) delta  =: alpha M_tot delta,  M_tot = M + w_B B + w_S S_r  with
w_B = beta/alpha, w_S = gamma/alpha  (KKT: w = 0 for an inactive constraint).  This is the generalised eigenproblem
A v = lambda M_tot v.  Whitening delta~ = M_tot^{1/2} delta turns it into the ordinary eigenproblem of
A~ = M_tot^{-1/2} A M_tot^{-1/2}; with r directions kept, W = M_tot^{-1/2} V~_r  (M_tot-orthonormal: W^T M_tot W = I) and
Lambda = W^T A W = diag(lambda_1..lambda_r).  Every feasible edit is delta = W c with ||c|| <= eps, and
   strengthen (signed by the mean relational displacement m_bar): maximise 2 b_bar^T c + c^T Lambda c, b_bar = W^T A m_bar
                -> boundary solution c_j = b_bar_j / (nu - lambda_j), nu > lambda_max with ||c|| = eps (secular equation);
                the edit is state-independent (an additive vector in the causal-metric-optimal subspace; r = 1 gives
                +-eps W_1, the top generalised eigenvector);
   suppress   (remove the token's own relational energy): minimise 2 b(h)^T c + c^T Lambda c, b(h) = W^T A (h - mu)
                -> c_j = -b_j / (lambda_j + nu), nu >= 0 with ||c|| <= eps  (nu = 0 when the unconstrained minimiser
                c = -Lambda^-1 b, i.e. whitened projection removal, already fits the ball).
The multipliers w_B, w_S are found by log-bisection so that the WORST-CASE constraint values over ||c|| <= eps,
eps^2 lambda_max(W^T B W) and eps sqrt(lambda_max(W^T S_r W)), meet their bounds at the unit dose eps_0
(eps_0 = median_i ||r_i||_N, so beta = 1 writes an edit as large, in nuisance units, as the relational DiD itself).

WHY THE METRIC MATTERS (the self-test's planted case)
------------------------------------------------------
If the DiD rows are r_i = rho_i u* + nu_i with nuisance leakage nu_i ~ Sigma_N, the Euclidean second moment R has top
eigenvector along the strongest nuisance axis whenever the leaked variance exceeds rho^2, whereas in whitened coordinates
the leakage is isotropic and R~ = M^{1/2} R M^{1/2} has top eigenvector M^{1/2} u*, so delta* = M^{-1/2} v~ recovers u*.

CONTROLS (same interface, same dose grid): ``euclidean_twin`` (Sigma_N = I sigma_bar^2), ``random_matched`` (random
M_tot-orthonormal r-frame, ||delta||_N = eps, same constraint weights), ``sham`` (eps = 0: the hook returns its input
unchanged, bit-identical), plus ``wrong_site`` / ``wrong_group`` from the steering table.

Interface: ``MinDistortionOperator.torch_fn(mode, beta, variant, device)`` returns ``fn(h)`` on ``[B, n_tokens, d]`` exactly
like ``steered_planner_ranking.TorchRunner._transform``; ``steered_planner_ranking.py --min-distortion-dir`` runs it as mode
``min_distortion``.  Files: ``<site>__s<step>__<group>.npz`` (+ ``.json``) from the ``fit`` sub-command.  ``--self-test``
runs the planted synthetic check (numpy) and, when torch is importable, the hook check on a fake predictor.
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
from protocol import NULL_CONTROL_HAZARD, OBJECT_HAZARD  # noqa: E402
from stats_utils import cluster_bootstrap_mean, finite, sign_flip_p  # noqa: E402

PROTOCOL = "cgs-causal-metric-steer-v0.1"
VARIANTS = ("main", "euclid", "random")
RANK_GRID = (1, 2, 4)
DEFAULT_WEIGHTS = {"scene": 1.0, "action": 1.0, "null": 1.0, "factors": 1.0, "token": 1.0}
INTERPRETATION_SCOPE = (
    "Donor-free additive/projective edit whose direction maximises the relational DiD energy measured with the nuisance-whitened "
    "(causal) inner product under a nuisance-metric norm budget, an action-subspace budget and a receiver-Mahalanobis budget. "
    "Fitted on discovery scenes only (full-data export; LOSO alignment reported as a diagnostic). Neither the metric nor the "
    "energy is a causal claim by itself: the claim comes from the steered-planner table (selectivity, controls, sealed confirmation)."
)


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #


def sym_power(S: np.ndarray, power: float, floor: float = 1e-12) -> np.ndarray:
    """S^power for symmetric PSD S via eigh (eigenvalues floored)."""
    w, U = np.linalg.eigh(0.5 * (S + S.T))
    w = np.clip(w, floor * max(float(w.max()), floor), None)
    return (U * w**power) @ U.T


def second_moment(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, np.float64)
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    if len(rows) == 0:
        return None
    return rows.T @ rows / len(rows)


def nuisance_covariance(groups: dict[str, np.ndarray], d: int, lam_rel: float, weights: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Sigma_N = sum_g w_g S_g + lambda I with S_g the second moment of group g's rows (groups weighted equally by default,
    so a group with many rows -- tokens -- does not swamp the factor contrasts)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    S = np.zeros((d, d))
    info: dict[str, Any] = {"groups": {}}
    total_w = 0.0
    for g, rows in groups.items():
        if rows is None or len(rows) == 0:
            continue
        Sg = second_moment(rows)
        if Sg is None:
            continue
        w = float(weights.get(g, 1.0))
        S += w * Sg
        total_w += w
        info["groups"][g] = {"n_rows": int(len(rows)), "weight": w, "trace": float(np.trace(Sg)), "top_eig_fraction": float(np.linalg.eigvalsh(Sg)[-1] / max(np.trace(Sg), 1e-300))}
    if total_w > 0:
        S /= total_w
    lam = lam_rel * float(np.trace(S)) / d if np.trace(S) > 0 else 1.0
    info.update({"lambda": lam, "lam_rel": lam_rel, "trace": float(np.trace(S)), "effective_rank": float(np.trace(S) ** 2 / max(np.sum(np.linalg.eigvalsh(S) ** 2), 1e-300))})
    return S + lam * np.eye(d), info


class Metric:
    """M = Sigma^-1 with its symmetric square roots; ``euclid`` uses the isotropic twin I / sigma_bar^2 (same trace)."""

    def __init__(self, Sigma: np.ndarray, euclid: bool = False) -> None:
        d = Sigma.shape[0]
        if euclid:
            s2 = float(np.trace(Sigma)) / d
            self.Sigma = s2 * np.eye(d)
        else:
            self.Sigma = 0.5 * (Sigma + Sigma.T)
        self.M = sym_power(self.Sigma, -1.0)
        self.d = d

    def norm(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float64))
        return np.sqrt(np.einsum("ij,jk,ik->i", x, self.M, x))


def solve_directions(A: np.ndarray, M_tot: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    """Top-``rank`` generalised eigenpairs of A v = lambda M_tot v via whitening.  Returns W [d, r] (M_tot-orthonormal),
    Lam [r], B_read = A W [d, r] (b(h) = B_read^T (h - mu)) and the whitened eigvecs Vt [d, r]."""
    Mi = sym_power(M_tot, -0.5)
    At = Mi @ A @ Mi
    w, V = np.linalg.eigh(0.5 * (At + At.T))
    order = np.argsort(w)[::-1][:rank]
    Vt = V[:, order]
    Lam = np.clip(w[order], 0.0, None)
    W = Mi @ Vt
    return {"W": W, "Lam": Lam, "B_read": A @ W, "Vt": Vt}


def strengthen_step(b: np.ndarray, Lam: np.ndarray, eps: float, sign_hint: float = 1.0) -> np.ndarray:
    """argmax 2 b^T c + c^T diag(Lam) c s.t. ||c|| <= eps (convex maximisation -> boundary, secular equation)."""
    b, Lam = np.asarray(b, np.float64), np.asarray(Lam, np.float64)
    if eps <= 0:
        return np.zeros_like(b)
    lmax = float(Lam.max()) if Lam.size else 0.0
    nb = float(np.linalg.norm(b))
    if nb < 1e-14 * max(eps, 1.0):
        c = np.zeros_like(b)
        c[int(np.argmax(Lam))] = eps * (1.0 if sign_hint >= 0 else -1.0)
        return c

    def phi(nu):
        return float(np.sum(b**2 / (nu - Lam) ** 2))

    lo, hi = lmax, lmax + nb / eps + 1e-12
    tiny = 1e-9 * max(lmax, 1.0)
    if phi(lo + tiny) < eps**2:  # hard case: b has no weight on the top direction; fill it up to the budget
        c = b / np.where(np.abs(lo - Lam) < tiny, np.inf, lo - Lam)
        j = int(np.argmax(Lam))
        c[j] = np.sqrt(max(eps**2 - float(np.sum(np.delete(c, j) ** 2)), 0.0)) * (1.0 if (b[j] >= 0 if abs(b[j]) > 0 else sign_hint >= 0) else -1.0)
        return c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) > eps**2:
            lo = mid
        else:
            hi = mid
    nu = hi
    return b / (nu - Lam)


def suppress_steps_numpy(Bp: np.ndarray, Lam: np.ndarray, eps: float) -> np.ndarray:
    """Rows b_i -> argmin 2 b^T c + c^T diag(Lam) c s.t. ||c|| <= eps (convex; c = -b/(Lam + nu))."""
    Bp, Lam = np.atleast_2d(np.asarray(Bp, np.float64)), np.asarray(Lam, np.float64)
    ok = Lam > 1e-12 * max(float(Lam.max()), 1e-300)
    c0 = np.where(ok[None, :], -Bp / np.where(ok, Lam, 1.0)[None, :], 0.0)
    n0 = np.linalg.norm(c0, axis=1)
    out = c0.copy()
    over = n0 > eps
    if eps <= 0:
        return np.zeros_like(Bp)
    for i in np.flatnonzero(over):
        b = Bp[i] * ok
        lo, hi = 0.0, float(np.linalg.norm(b)) / eps + 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.sum(b**2 / (Lam + mid) ** 2) > eps**2:
                lo = mid
            else:
                hi = mid
        out[i] = -b / (Lam + hi)
    return out


def _worst_case(W: np.ndarray, Q: np.ndarray, eps0: float) -> float:
    """max_{||c|| <= eps0} c^T W^T Q W c = eps0^2 lambda_max(W^T Q W)."""
    G = W.T @ Q @ W
    return float(eps0**2 * np.linalg.eigvalsh(0.5 * (G + G.T))[-1]) if G.size else 0.0


def fit_directions_constrained(A: np.ndarray, M: np.ndarray, rank: int, eps0: float, B_act: np.ndarray | None, tau2: float | None,
                               S_r: np.ndarray | None, incr: float | None, max_rounds: int = 3) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Find w_B, w_S (KKT multiplier ratios) by log-bisection so that the worst-case D_action and D_local increment over the
    dose ball ||c|| <= eps0 meet their bounds; returns the directions under M_tot and the constraint diagnostics."""
    d = A.shape[0]
    scale = float(np.trace(M)) / d
    w = {"B": 0.0, "S": 0.0}
    cons = [("B", B_act, tau2), ("S", S_r, (incr**2 if incr is not None else None))]

    def solve(wB, wS):
        Mt = M + (wB * scale) * (B_act if B_act is not None else 0.0) + (wS * scale) * (S_r if S_r is not None else 0.0)
        return solve_directions(A, Mt, rank), Mt

    dirs, Mt = solve(0.0, 0.0)
    for _ in range(max_rounds):
        changed = False
        for key, Q, bound in cons:
            if Q is None or bound is None:
                continue
            val = _worst_case(dirs["W"], Q, eps0)
            if val <= bound:
                continue
            changed = True
            lo, hi = 0.0, 1e-3
            for _k in range(60):
                dd, _ = solve(hi if key == "B" else w["B"], hi if key == "S" else w["S"])
                if _worst_case(dd["W"], Q, eps0) <= bound:
                    break
                lo, hi = hi, hi * 2.0
            for _k in range(40):
                mid = 0.5 * (lo + hi)
                dd, _ = solve(mid if key == "B" else w["B"], mid if key == "S" else w["S"])
                if _worst_case(dd["W"], Q, eps0) <= bound:
                    hi = mid
                else:
                    lo = mid
            w[key] = hi
            dirs, Mt = solve(w["B"], w["S"])
        if not changed:
            break
    diag = {"w_B": w["B"], "w_S": w["S"], "metric_scale": scale,
            "worst_case_D_action": (_worst_case(dirs["W"], B_act, eps0) if B_act is not None else None), "tau2": tau2,
            "worst_case_mahal_increment": (float(np.sqrt(_worst_case(dirs["W"], S_r, eps0))) if S_r is not None else None), "mahal_incr_bound": incr}
    dirs["M_tot"] = Mt
    return dirs, diag


def random_frame(Mi_half: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(Mi_half.shape[0], rank)))
    return Mi_half @ Q


# --------------------------------------------------------------------------- #
# Operator object (numpy reference + torch hook)
# --------------------------------------------------------------------------- #


class MinDistortionOperator:
    """Per-site operator: ``mean`` [d]; per variant ``W`` [d, r], ``B_read`` [d, r], ``Lam`` [r], ``c_star`` [r] (unit-eps
    strengthen step), ``eps0``; ``receiver_mean`` / ``receiver_var`` [d] for the D_local report."""

    def __init__(self, arrays: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        self.a = arrays
        self.meta = meta or {}
        self.d = int(arrays["mean"].shape[0])

    # ---- io ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def load(path: Path) -> "MinDistortionOperator":
        path = Path(path)
        with np.load(path) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return MinDistortionOperator(arrays, meta)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(finite(self.meta)), **{k: np.asarray(v) for k, v in self.a.items()})
        return path

    def variant_key(self, variant: str) -> str:
        return {"main": "main", "min_distortion": "main", "euclidean_twin": "euclid", "euclid": "euclid", "random_matched": "random", "random": "random"}[variant]

    # ---- numpy reference (tests / diagnostics) -------------------------------------------------------------------------
    def apply_numpy(self, H: np.ndarray, mode: str, beta: float, variant: str = "main") -> np.ndarray:
        v = self.variant_key(variant)
        H = np.atleast_2d(np.asarray(H, np.float64))
        if beta <= 0:
            return H.copy()
        W, Lam, Br, mu = self.a[f"{v}_W"], self.a[f"{v}_Lam"], self.a[f"{v}_B_read"], self.a["mean"]
        eps = beta * float(self.a[f"{v}_eps0"])
        if mode == "strengthen" or v == "random":
            c = eps * np.asarray(self.a[f"{v}_c_star"]) * (1.0 if mode == "strengthen" else -1.0)
            return H + c @ W.T
        if mode == "suppress":
            C = suppress_steps_numpy((H - mu) @ Br, Lam, eps)
            return H + C @ W.T
        raise ValueError(f"unknown mode {mode}")

    def energy_numpy(self, H: np.ndarray) -> np.ndarray:
        """E_N(h - mu) = (h - mu)^T A (h - mu) evaluated through the stored main directions' B_read only when A is absent:
        here A is stored as ``A_rel`` when small enough, else the r-dim restriction is used."""
        H = np.atleast_2d(np.asarray(H, np.float64)) - self.a["mean"]
        if "A_rel" in self.a:
            return np.einsum("ij,jk,ik->i", H, self.a["A_rel"], H)
        p = H @ self.a["main_Vt_M"]  # whitened relational coordinates
        return np.sum(p**2 * self.a["main_Lam"], axis=1)

    # ---- torch hook ---------------------------------------------------------------------------------------------------
    def torch_fn(self, mode: str, beta: float, variant: str, device: Any, sign: float | None = None):
        """``fn(h)`` for ``steered_planner_ranking.PredictorSteerer``: h [B, n, d] -> edited h (same dtype)."""
        import torch

        v = self.variant_key(variant)
        if beta <= 0:  # sham: bit-identical
            return lambda h: h
        eps = float(beta * float(self.a[f"{v}_eps0"]))
        W = torch.as_tensor(self.a[f"{v}_W"], dtype=torch.float32, device=device)  # [d, r]
        Br = torch.as_tensor(self.a[f"{v}_B_read"], dtype=torch.float32, device=device)
        Lam = torch.as_tensor(self.a[f"{v}_Lam"], dtype=torch.float32, device=device)
        mu = torch.as_tensor(self.a["mean"], dtype=torch.float32, device=device)
        c_star = torch.as_tensor(self.a[f"{v}_c_star"], dtype=torch.float32, device=device)
        s = float(sign if sign is not None else (1.0 if mode == "strengthen" else -1.0))
        if mode == "strengthen" or v == "random":
            delta = (s * eps) * (c_star @ W.T)  # [d]

            def fn(h):
                return (h.float() + delta.reshape(*([1] * (h.ndim - 1)), -1)).to(h.dtype)

            return fn
        if mode != "suppress":
            raise ValueError(f"unknown mode {mode}")
        ok = Lam > 1e-12 * float(Lam.max().clamp_min(1e-30))
        Lam_safe = torch.where(ok, Lam, torch.ones_like(Lam))

        def fn(h):
            hf = h.float()
            shape = hf.shape
            x = (hf - mu).reshape(-1, shape[-1])
            b = (x @ Br) * ok  # [n, r]
            c0 = -b / Lam_safe
            n0 = c0.norm(dim=1)
            over = n0 > eps
            c = c0.clone()
            if bool(over.any()):
                bo = b[over]
                lo = torch.zeros(bo.shape[0], device=bo.device)
                hi = bo.norm(dim=1) / eps + 1e-12
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    val = (bo**2 / (Lam + mid[:, None]) ** 2).sum(dim=1)
                    big = val > eps**2
                    lo = torch.where(big, mid, lo)
                    hi = torch.where(big, hi, mid)
                c[over] = -bo / (Lam + hi[:, None])
            return (hf + (c @ W.T).reshape(shape)).to(h.dtype)

        return fn


# --------------------------------------------------------------------------- #
# Fitting from rows (arm-agnostic; used by the real-data driver and the self-test)
# --------------------------------------------------------------------------- #


def fit_operator(rel_rows: np.ndarray, nuisance_groups: dict[str, np.ndarray], action_rows: np.ndarray | None, receiver_tokens: np.ndarray | None,
                 mean: np.ndarray, *, rank: int = 4, k_action: int = 4, lam_rel: float = 0.1, tau_action_rel: float = 0.1, mahal_incr: float = 0.25,
                 weights: dict[str, float] | None = None, seed: int = 0, loso_scene: np.ndarray | None = None, loso: bool = True,
                 store_A: bool = True) -> MinDistortionOperator:
    """Fit the three variants (main causal metric, Euclidean twin, matched random frame) on the given rows."""
    rng = np.random.default_rng(seed)
    d = int(mean.shape[0])
    rel = np.asarray(rel_rows, np.float64)
    rel = rel[np.all(np.isfinite(rel), axis=1)]
    if len(rel) < 3:
        raise ValueError(f"need >= 3 relational rows, got {len(rel)}")
    Sigma, sig_info = nuisance_covariance(nuisance_groups, d, lam_rel, weights)
    # action subspace + relational rows projected off it (geometry_cross_arm.relational_rows convention)
    Q_act = None
    if action_rows is not None and k_action > 0:
        act = np.asarray(action_rows, np.float64)
        act = act[np.all(np.isfinite(act), axis=1)]
        if len(act) >= 1:
            from sonar_metrics import svd_subspace

            Q_act = svd_subspace(act, k_action)
            rel = rel - (rel @ Q_act) @ Q_act.T
    R = rel.T @ rel / len(rel)
    m_bar = rel.mean(0)
    B_act = Q_act @ Q_act.T if Q_act is not None else None
    tau2 = None
    if Q_act is not None:
        act_proj = np.linalg.norm(act @ Q_act, axis=1)
        tau2 = float((tau_action_rel * np.median(act_proj)) ** 2)
    S_r, rec_mean, rec_var = None, None, None
    if receiver_tokens is not None and len(receiver_tokens) >= 8:
        rt = np.asarray(receiver_tokens, np.float64)
        rec_mean, rec_var = rt.mean(0), rt.var(0) + 1e-8
        S_r = np.diag(1.0 / rec_var) / d
    arrays: dict[str, Any] = {"mean": np.asarray(mean, np.float64), "m_bar": m_bar}
    if Q_act is not None:
        arrays["Q_act"] = Q_act
    if rec_mean is not None:
        arrays["receiver_mean"], arrays["receiver_var"] = rec_mean, rec_var
    meta: dict[str, Any] = {"protocol": PROTOCOL, "d": d, "rank": rank, "k_action": k_action, "lam_rel": lam_rel, "tau_action_rel": tau_action_rel, "mahal_incr": mahal_incr,
                            "n_rel_rows": int(len(rel)), "nuisance": sig_info, "tau2": tau2, "variants": {}}
    metrics = {"main": Metric(Sigma), "euclid": Metric(Sigma, euclid=True)}
    main_Mtot = None
    for v, met in metrics.items():
        A = met.M @ R @ met.M
        eps0 = float(np.median(met.norm(rel)))
        dirs, cdiag = fit_directions_constrained(A, met.M, rank, eps0, B_act, tau2, S_r, mahal_incr)
        if v == "main":
            main_Mtot = dirs["M_tot"]
        b_bar = dirs["B_read"].T @ m_bar
        c_star = strengthen_step(b_bar, dirs["Lam"], 1.0)  # unit eps; scaled at runtime
        arrays[f"{v}_W"], arrays[f"{v}_B_read"], arrays[f"{v}_Lam"], arrays[f"{v}_c_star"], arrays[f"{v}_eps0"] = dirs["W"], dirs["B_read"], dirs["Lam"], c_star, np.float64(eps0)
        arrays[f"{v}_Vt_M"] = sym_power(dirs["M_tot"], 0.5) @ dirs["Vt"] if v == "main" else np.zeros((d, 0))
        delta1 = eps0 * (c_star @ dirs["W"].T)
        meta["variants"][v] = {
            "eps0": eps0, "Lam": [float(x) for x in dirs["Lam"]], "c_star_unit": [float(x) for x in c_star], "constraints": cdiag,
            "unit_edit": {"euclidean_norm": float(np.linalg.norm(delta1)), "causal_norm": float(metrics["main"].norm(delta1)[0]),
                          "cos_with_mean_rel": float(delta1 @ m_bar / max(np.linalg.norm(delta1) * np.linalg.norm(m_bar), 1e-300)),
                          "D_action": (float(delta1 @ B_act @ delta1) if B_act is not None else None),
                          "mahal_increment": (float(np.sqrt(delta1 @ S_r @ delta1)) if S_r is not None else None),
                          "relational_energy_gain_causal": float(delta1 @ (metrics["main"].M @ R @ metrics["main"].M) @ delta1),
                          "relational_energy_gain_euclid": float(delta1 @ R @ delta1)},
        }
    # matched random frame under the main constraint weights (M_tot-orthonormal, ||delta||_N = eps)
    Wr = random_frame(sym_power(main_Mtot, -0.5), rank, rng)
    A_main = metrics["main"].M @ R @ metrics["main"].M
    e = rng.normal(size=rank)
    e /= np.linalg.norm(e)
    arrays["random_W"], arrays["random_B_read"], arrays["random_Lam"], arrays["random_c_star"], arrays["random_eps0"] = Wr, A_main @ Wr, np.clip(np.diag(Wr.T @ A_main @ Wr), 0, None), e, arrays["main_eps0"]
    arrays["random_Vt_M"] = np.zeros((d, 0))
    delta_r = float(arrays["main_eps0"]) * (e @ Wr.T)
    meta["variants"]["random"] = {"eps0": float(arrays["main_eps0"]), "unit_edit": {"euclidean_norm": float(np.linalg.norm(delta_r)), "causal_norm": float(metrics["main"].norm(delta_r)[0]),
                                                                                    "cos_with_mean_rel": float(delta_r @ m_bar / max(np.linalg.norm(delta_r) * np.linalg.norm(m_bar), 1e-300)),
                                                                                    "D_action": (float(delta_r @ B_act @ delta_r) if B_act is not None else None),
                                                                                    "mahal_increment": (float(np.sqrt(delta_r @ S_r @ delta_r)) if S_r is not None else None),
                                                                                    "relational_energy_gain_causal": float(delta_r @ A_main @ delta_r)}}
    if store_A:
        arrays["A_rel"] = A_main
    # direction-level comparison of the twins
    w_main, w_euc = arrays["main_W"][:, 0], arrays["euclid_W"][:, 0]
    meta["twin_comparison"] = {"cos_main_vs_euclid_top": float(abs(w_main @ w_euc) / max(np.linalg.norm(w_main) * np.linalg.norm(w_euc), 1e-300)),
                               "cos_main_top_vs_mean_rel": float(abs(w_main @ m_bar) / max(np.linalg.norm(w_main) * np.linalg.norm(m_bar), 1e-300)),
                               "cos_euclid_top_vs_mean_rel": float(abs(w_euc @ m_bar) / max(np.linalg.norm(w_euc) * np.linalg.norm(m_bar), 1e-300))}
    if loso and loso_scene is not None:
        meta["loso"] = loso_alignment(rel_rows, nuisance_groups, loso_scene, d, lam_rel, weights, seed)
    return MinDistortionOperator(arrays, meta)


def loso_alignment(rel_rows: np.ndarray, nuisance_groups: dict[str, np.ndarray], scene_of: dict[str, np.ndarray] | np.ndarray, d: int, lam_rel: float,
                   weights: dict[str, float] | None, seed: int, n_perm: int = 5000) -> dict[str, Any]:
    """Leave-one-scene-out: refit Sigma_N (rows of the other scenes) and the rank-1 direction on the other scenes' relational
    rows; score the held-out row by its causal-metric cosine with the held-out direction (main) and the Euclidean-twin
    cosine (euclid).  ``scene_of``: dict group -> scene label per row (``rel`` for the relational rows)."""
    rel = np.asarray(rel_rows, np.float64)
    if isinstance(scene_of, np.ndarray):
        scene_of = {"rel": scene_of}
    sc_rel = np.asarray(scene_of["rel"])
    scenes = np.unique(sc_rel)
    out = {"main": [], "euclid": []}
    for s in scenes:
        te = sc_rel == s
        groups = {}
        for g, rows in nuisance_groups.items():
            if rows is None or len(rows) == 0:
                continue
            lab = scene_of.get(g)
            groups[g] = rows if lab is None else rows[np.asarray(lab) != s]
        Sigma, _ = nuisance_covariance(groups, d, lam_rel, weights)
        tr = rel[~te]
        if len(tr) < 2 or not np.all(np.isfinite(rel[te])):
            continue
        for v, met in (("main", Metric(Sigma)), ("euclid", Metric(Sigma, euclid=True))):
            A = met.M @ (tr.T @ tr / len(tr)) @ met.M
            w = solve_directions(A, met.M, 1)["W"][:, 0]
            sign = 1.0 if (tr.mean(0) @ met.M @ w) >= 0 else -1.0
            for r in rel[te]:
                out[v].append(float(sign * (r @ met.M @ w) / max(met.norm(r)[0] * np.sqrt(w @ met.M @ w), 1e-300)))
    res = {}
    for v, vals in out.items():
        x = np.asarray(vals)
        res[v] = {**cluster_bootstrap_mean(x, n_boot=1000, seed=seed), "sign_flip_p": sign_flip_p(x, n_perm=n_perm, seed=seed), "t": (float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) > 1e-12 else None),
                  "abs_mean": float(np.mean(np.abs(x))) if len(x) else None}
    res["definition"] = ("held-out cosine, in the metric of the variant, between the held-out relational row and the rank-1 direction fitted on the other scenes "
                         "(Sigma_N refitted without the held-out scene); sign of the direction from the training-fold mean, so under the null of no consistent "
                         "direction the signed statistic is negatively biased (leave-one-out) -- a positive mean is the criterion, abs_mean is the unsigned capture")
    return res


# --------------------------------------------------------------------------- #
# Real-data rows from a localization dump (single arm)
# --------------------------------------------------------------------------- #


FACTOR_KEYS = ("hazard_dist_m", "prefix_throttle", "hazard_lateral_offset_m")


def arm_cells(dump: Path, discovery_seeds: Path | None, solid: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    from geometry_cross_arm import load_arm

    arm = load_arm(dump, discovery_seeds)
    by: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for c in arm["cells"]:
        by.setdefault(str(c["pair_id"]), {})[(int(c["hazard"]), int(c["candidate_action"]))] = c
    need = {(0, 0), (0, 1), (solid, 0), (solid, 1)}
    scenes = sorted(s for s, ks in by.items() if need <= set(ks))
    cells = [dict(c, row=int(c["row"])) for s in scenes for c in by[s].values()]
    return arm, cells, scenes


def manifest_factors(stimulus: Path, cells: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per scene: the v0.9 randomised factor values (identical across a scene's cells); empty when absent."""
    mp = stimulus / "manifest.jsonl"
    if not mp.exists():
        return {}
    rows = {}
    for line in mp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["cell_id"]] = r
    out: dict[str, dict[str, float]] = {}
    for c in cells:
        r = rows.get(c["cell_id"])
        if r is None:
            continue
        vals = {k: float(r[k]) for k in FACTOR_KEYS if r.get(k) is not None and np.isfinite(float(r[k]))}
        if vals:
            out.setdefault(str(c["pair_id"]), {}).update(vals)
    return out


def build_rows(P: np.ndarray, cells: list[dict[str, Any]], scenes: list[str], solid: int, factors: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Relational rows (DiD_solid - DiD_null where H0' exists), hazard-free action rows, and the nuisance groups from the
    group-pooled cell vectors ``P[row]``."""
    by: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for c in cells:
        by.setdefault(str(c["pair_id"]), {})[(int(c["hazard"]), int(c["candidate_action"]))] = P[c["row"]]
    d = P.shape[1]
    NULL = NULL_CONTROL_HAZARD
    rel, act, null_disp, scene_rows_, scene_lab = [], [], [], {}, {"rel": [], "action": [], "null": [], "scene": []}
    for s in scenes:
        h = by[s]
        base = h[(0, 1)] - h[(0, 0)]
        did = (h[(solid, 1)] - h[(solid, 0)]) - base
        if (NULL, 0) in h and (NULL, 1) in h:
            did = did - ((h[(NULL, 1)] - h[(NULL, 0)]) - base)
        rel.append(did)
        scene_lab["rel"].append(s)
        levels = [lv for lv in (0, NULL) if (lv, 0) in h and (lv, 1) in h]
        act.append(np.mean([h[(lv, 1)] - h[(lv, 0)] for lv in levels], axis=0))
        scene_lab["action"].append(s)
        for a in (0, 1):
            if (NULL, a) in h:
                null_disp.append(h[(NULL, a)] - h[(0, a)])
                scene_lab["null"].append(s)
        for lv in (0, NULL):
            for a in (0, 1):
                if (lv, a) in h:
                    scene_rows_.setdefault((lv, a), []).append((s, h[(lv, a)]))
    scene_var, fac_rows = [], []
    fac_info: dict[str, Any] = {"used": False}
    for key, items in scene_rows_.items():
        X = np.asarray([v for _, v in items])
        if len(X) >= 3:
            Xc = X - X.mean(0)
            scene_var.extend(Xc)
            scene_lab["scene"].extend(s for s, _ in items)
            if factors:
                F = np.asarray([[factors.get(s, {}).get(k, np.nan) for k in FACTOR_KEYS] for s, _ in items], np.float64)
                keep = np.all(np.isfinite(F), axis=0) & (np.nanstd(F, axis=0) > 1e-6)
                if keep.any() and len(X) >= 4:
                    Z = (F[:, keep] - F[:, keep].mean(0)) / F[:, keep].std(0)
                    lam = 1.0
                    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Xc)  # [k, d]: displacement per 1 SD
                    fac_rows.extend(beta)
                    fac_info = {"used": True, "factors": [k for k, m in zip(FACTOR_KEYS, keep) if m], "sd": {k: float(F[:, i].std()) for i, k in enumerate(FACTOR_KEYS) if keep[i]},
                                "r2": float(1 - np.sum((Xc - Z @ beta) ** 2) / max(np.sum(Xc**2), 1e-300))}
    groups = {"scene": np.asarray(scene_var).reshape(-1, d) if scene_var else np.zeros((0, d)), "action": np.asarray(act), "null": np.asarray(null_disp).reshape(-1, d) if null_disp else np.zeros((0, d)),
              "factors": np.asarray(fac_rows).reshape(-1, d) if fac_rows else np.zeros((0, d))}
    return {"rel": np.asarray(rel), "action": np.asarray(act), "groups": groups, "scene_of": {k: np.asarray(v) for k, v in scene_lab.items()}, "factors": fac_info}


def token_nuisance_rows(acts: np.ndarray, tix: np.ndarray, cells: list[dict[str, Any]], token_sets, step: int, group: str, levels: tuple[int, ...], max_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows, lab = [], []
    for c in cells:
        if int(c["hazard"]) not in levels:
            continue
        r = c["row"]
        ti = tix[r, step]
        want = token_sets[(str(c["pair_id"]), step, group)]
        m = np.isin(ti, want) & (ti >= 0)
        if m.sum() >= 2:
            X = acts[r, step, m].astype(np.float64)
            rows.extend(X - X.mean(0))
            lab.extend([str(c["pair_id"])] * int(m.sum()))
    if not rows:
        return np.zeros((0, acts.shape[-1])), np.zeros(0)
    rows, lab = np.asarray(rows), np.asarray(lab)
    if len(rows) > max_rows:
        keep = np.sort(rng.choice(len(rows), size=max_rows, replace=False))
        rows, lab = rows[keep], lab[keep]
    return rows, lab


def receiver_token_rows(acts: np.ndarray, tix: np.ndarray, cells: list[dict[str, Any]], token_sets, step: int, group: str, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    rows = []
    for c in cells:
        r = c["row"]
        ti = tix[r, step]
        want = token_sets[(str(c["pair_id"]), step, group)]
        m = np.isin(ti, want) & (ti >= 0)
        if m.any():
            rows.extend(acts[r, step, m].astype(np.float64))
    rows = np.asarray(rows).reshape(-1, acts.shape[-1])
    if len(rows) > max_rows:
        rows = rows[np.sort(rng.choice(len(rows), size=max_rows, replace=False))]
    return rows


def fit_from_dump(args: argparse.Namespace) -> dict[str, Any]:
    from geometry_cross_arm import load_site, token_sets_for
    from geometry_localize import pool_site
    from token_groups import set_domain

    t0 = time.time()
    set_domain(args.domain)
    rng = np.random.default_rng(args.seed)
    arm, cells, scenes = arm_cells(args.dump, args.discovery_seeds, args.solid_level)
    if len(scenes) < 4:
        raise SystemExit(f"only {len(scenes)} complete scenes for levels {{0, {args.solid_level}}} x actions (need >= 4)")
    n_steps = arm["n_steps"]
    token_sets = token_sets_for(args.stimulus, cells, tuple(args.groups), n_steps, arm["offset"], args.domain)
    rows_idx = np.asarray([c["row"] for c in cells])
    cells_local = [dict(c, row=i) for i, c in enumerate(cells)]  # rows re-indexed after load_site(rows)
    factors = manifest_factors(args.stimulus, cells) if args.manifest_factors else {}
    sites = list(args.sites) if args.sites else sorted(p.stem for p in (args.dump / "activations").glob("*.npz") if "adaln" not in p.stem)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for si, site in enumerate(sites):
        acts, tix = load_site(args.dump, site, rows_idx)
        if acts.shape[2] == 1:
            entries.append({"site_id": site, "status": "adaln_skipped"})
            continue
        for step in [s for s in args.steps if s < acts.shape[1]]:
            for g in args.groups:
                entry: dict[str, Any] = {"site_id": site, "step": step, "group": g}
                P, _ = pool_site(acts, tix, cells_local, token_sets, step, g)
                ok = np.all(np.isfinite(P), axis=1)
                if ok.sum() < len(cells_local):
                    bad = {str(c["pair_id"]) for c in cells_local if not ok[c["row"]]}
                    keep = [s for s in scenes if s not in bad]
                    sub = [c for c in cells_local if str(c["pair_id"]) in set(keep)]
                    entry["n_scenes_dropped_no_tokens"] = len(scenes) - len(keep)
                else:
                    keep, sub = scenes, cells_local
                if len(keep) < 4:
                    entry["status"] = "no_tokens"
                    entries.append(entry)
                    continue
                rows = build_rows(P, sub, keep, args.solid_level, factors)
                groups = dict(rows["groups"])
                scene_of = dict(rows["scene_of"])
                if args.token_nuisance:
                    tok, lab = token_nuisance_rows(acts, tix, sub, token_sets, step, g, (0, NULL_CONTROL_HAZARD), args.max_token_rows, rng)
                    groups["token"], scene_of["token"] = tok, lab
                recv = receiver_token_rows(acts, tix, sub, token_sets, step, g, args.max_token_rows, rng)
                mean = P[[c["row"] for c in sub]].mean(0)
                op = fit_operator(rows["rel"], groups, rows["action"], recv, mean, rank=args.rank, k_action=args.k_action, lam_rel=args.lam_rel, tau_action_rel=args.tau_action_rel,
                                  mahal_incr=args.mahal_incr, seed=args.seed, loso_scene=scene_of, loso=not args.no_loso, store_A=(P.shape[1] <= 1024))
                op.meta.update({"site": site, "step": step, "group": g, "solid_level": args.solid_level, "n_scenes": len(keep), "scenes": keep, "dump": str(args.dump), "stimulus": str(args.stimulus),
                                "discovery_seeds": str(args.discovery_seeds), "factors": rows["factors"], "token_nuisance": bool(args.token_nuisance), "fitted_on": "discovery scenes only (full-data export)"})
                path = op.save(args.out / f"{site}__s{step}__{g}.npz")
                (args.out / f"{site}__s{step}__{g}.json").write_text(json.dumps(finite(op.meta), indent=1) + "\n")
                entry.update({"status": "ok", "file": str(path), "n_scenes": len(keep), "variants": op.meta["variants"], "twin_comparison": op.meta["twin_comparison"], "loso": op.meta.get("loso"),
                              "nuisance": op.meta["nuisance"], "factors": rows["factors"]})
                entries.append(entry)
        print(f"[{si + 1}/{len(sites)}] {site} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    report = {"protocol": PROTOCOL, "dump": str(args.dump), "stimulus": str(args.stimulus), "discovery_seeds": str(args.discovery_seeds), "solid_level": args.solid_level, "n_scenes": len(scenes),
              "scenes": scenes, "sites": sites, "steps": args.steps, "groups": args.groups, "rank": args.rank, "k_action": args.k_action, "lam_rel": args.lam_rel, "tau_action_rel": args.tau_action_rel,
              "mahal_incr": args.mahal_incr, "token_nuisance": bool(args.token_nuisance), "entries": entries, "runtime_s": time.time() - t0, "interpretation_scope": INTERPRETATION_SCOPE}
    (args.out / "causal_metric_fit.json").write_text(json.dumps(finite(report), indent=1) + "\n")
    return report


# --------------------------------------------------------------------------- #
# Optional pullback check with the existing forward-AD JVP (model needed)
# --------------------------------------------------------------------------- #


def pullback_check(args: argparse.Namespace) -> dict[str, Any]:
    """For every operator file: JVP of the step-k route-token prediction along each direction w_j (unit eps_0 edit written
    on the group tokens) on the HAZARD-FREE cells, relative to the JVP along the hazard-free action row; writes
    ``pullback.json`` and a per-file ``eps_cap`` (dose cap so that the ratio stays <= --tau-pullback)."""
    import torch

    from action_jacobian_sonar import downstream_response
    from localize_interaction import NULL_CELLS, encode_cell, merge_manifests, select_pairs
    from model_action_sensitivity import load_model
    from predictor_hooks import predictor_of, spatial_tokens_of
    from token_groups import frame_for_step, load_cell_groups, set_domain, union

    set_domain(args.domain)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wm, _ = load_model(args.repo, args.config, args.checkpoint, args.model_name, str(device))
    predictor, n_spatial = predictor_of(wm), spatial_tokens_of(wm)
    pairs = select_pairs(merge_manifests(args.artifacts), args.seeds_file, True)
    pids = sorted(pairs)[: args.max_scenes]
    files = sorted(args.operator_dir.glob("*.npz"))
    out: dict[str, Any] = {"protocol": PROTOCOL, "files": {}, "tau_pullback": args.tau_pullback, "n_scenes": len(pids)}
    for f in files:
        op = MinDistortionOperator.load(f)
        site, step, group = op.meta["site"], int(op.meta["step"]), op.meta["group"]
        ratios = {v: [] for v in VARIANTS}
        for pid in pids:
            cells = pairs[pid]
            for key in [(0, 0), (0, 1)] + [k for k in NULL_CELLS if k in cells]:
                lat = encode_cell(wm, *cells[key], device)
                grp = load_cell_groups(*cells[key])
                idx = np.asarray(union(grp.group(frame_for_step(step, 0), group)), dtype=int)
                if len(idx) == 0:
                    continue
                z, a = lat["z_context"], lat["actions"]
                # action row at the site is unknown here; use the operator's stored Q_act top direction as the reference tangent
                ref = op.a["Q_act"][:, 0] if "Q_act" in op.a else op.a["m_bar"] / np.linalg.norm(op.a["m_bar"])
                ref_t = torch.zeros(n_spatial, op.d, device=device)
                ref_t[torch.as_tensor(idx, device=device)] = torch.as_tensor(ref, dtype=torch.float32, device=device)
                ref_resp = downstream_response(wm, predictor, z, a, step, n_spatial, site, ref_t).flatten().norm().item()
                for v in VARIANTS:
                    W, c = op.a[f"{v}_W"], op.a[f"{v}_c_star"]
                    delta = float(op.a[f"{v}_eps0"]) * (c @ W.T)
                    t = torch.zeros(n_spatial, op.d, device=device)
                    t[torch.as_tensor(idx, device=device)] = torch.as_tensor(delta, dtype=torch.float32, device=device)
                    resp = downstream_response(wm, predictor, z, a, step, n_spatial, site, t).flatten().norm().item()
                    ratios[v].append(resp / max(ref_resp * float(np.linalg.norm(delta)) / max(np.linalg.norm(ref), 1e-12), 1e-12))
        summ = {v: cluster_bootstrap_mean(np.asarray(r), n_boot=500, seed=args.seed) for v, r in ratios.items() if r}
        caps = {v: float(min(1.0, args.tau_pullback / max(s["mean"], 1e-12))) for v, s in summ.items()}
        out["files"][f.name] = {"site": site, "step": step, "group": group, "pullback_ratio_per_unit_norm": summ, "eps_cap_beta": caps}
        op.meta["pullback"] = out["files"][f.name]
        op.save(f)
    (args.operator_dir / "pullback.json").write_text(json.dumps(finite(out), indent=1) + "\n")
    return out


# --------------------------------------------------------------------------- #
# Self-test: planted relational direction inside a nuisance-dominated covariance
# --------------------------------------------------------------------------- #


def synthetic_rows(d: int = 64, n_scenes: int = 24, n_nuis: int = 6, rho: float = 1.0, leak: float = 1.0, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, n_nuis + 2)))
    N_dirs, u_star, a_dir = Q[:, :n_nuis], Q[:, n_nuis], Q[:, n_nuis + 1]
    sig = np.linspace(3.0, 1.0, n_nuis)
    Sigma_true = (N_dirs * sig**2) @ N_dirs.T + 0.05 * np.eye(d)

    def nuis(n):
        return rng.normal(size=(n, n_nuis)) * sig @ N_dirs.T + 0.05**0.5 * rng.normal(size=(n, d))

    rel = rho * (1.0 + 0.2 * rng.normal(size=(n_scenes, 1))) * u_star[None, :] + leak * nuis(n_scenes)  # relational DiD rows with nuisance leakage
    scene = np.array([f"s{i:02d}" for i in range(n_scenes)])
    groups = {"scene": nuis(4 * n_scenes), "action": 2.0 * a_dir[None, :] * (1 + 0.1 * rng.normal(size=(n_scenes, 1))) + 0.3 * nuis(n_scenes), "null": 0.8 * nuis(2 * n_scenes)}
    scene_of = {"rel": scene, "scene": np.repeat(scene, 4), "action": scene, "null": np.repeat(scene, 2)}
    recv = nuis(400) + 0.5 * rng.normal(size=(400, d))
    return {"rel": rel, "groups": groups, "scene_of": scene_of, "receiver": recv, "mean": np.zeros(d), "u_star": u_star, "a_dir": a_dir, "Sigma_true": Sigma_true, "N_dirs": N_dirs}


def self_test(out: Path | None = None, seed: int = 0) -> dict[str, Any]:
    t0 = time.time()
    syn = synthetic_rows(seed=seed)
    op = fit_operator(syn["rel"], syn["groups"], syn["groups"]["action"], syn["receiver"], syn["mean"], rank=2, k_action=1, seed=seed, loso_scene=syn["scene_of"])
    u = syn["u_star"]
    cosu = lambda w: float(abs(w @ u) / max(np.linalg.norm(w) * np.linalg.norm(u), 1e-300))  # noqa: E731
    d_main = op.a["main_eps0"] * (op.a["main_c_star"] @ op.a["main_W"].T)
    d_euc = op.a["euclid_eps0"] * (op.a["euclid_c_star"] @ op.a["euclid_W"].T)
    d_rand = op.a["random_eps0"] * (op.a["random_c_star"] @ op.a["random_W"].T)
    checks: dict[str, Any] = {
        "cos_main_edit_vs_planted": cosu(d_main), "cos_euclid_edit_vs_planted": cosu(d_euc), "cos_random_edit_vs_planted": cosu(d_rand),
        "cos_main_edit_vs_top_nuisance": float(abs(d_main @ syn["N_dirs"][:, 0]) / np.linalg.norm(d_main)), "cos_euclid_edit_vs_top_nuisance": float(abs(d_euc @ syn["N_dirs"][:, 0]) / np.linalg.norm(d_euc)),
        "signed_main_edit_on_planted": float(d_main @ u), "D_action_main": op.meta["variants"]["main"]["unit_edit"]["D_action"], "tau2": op.meta["tau2"],
        "mahal_incr_main": op.meta["variants"]["main"]["unit_edit"]["mahal_increment"], "mahal_incr_bound": op.meta["mahal_incr"],
        "loso": op.meta.get("loso"),
    }
    # numpy application: sham identity, strengthen adds the edit, suppress removes whitened relational energy, random ~ orthogonal
    H = np.random.default_rng(seed + 1).normal(size=(5, syn["mean"].shape[0])) + 2.0 * u[None, :]
    checks["sham_identical"] = bool(np.array_equal(op.apply_numpy(H, "strengthen", 0.0), H))
    Hs = op.apply_numpy(H, "strengthen", 1.0)
    checks["strengthen_matches_closed_form"] = bool(np.allclose(Hs - H, d_main[None, :]))
    Hsup = op.apply_numpy(H, "suppress", 1.0)
    e_before, e_after = op.energy_numpy(H), op.energy_numpy(Hsup)
    checks["suppress_reduces_causal_energy"] = bool(np.all(e_after < e_before))
    Lam = op.a["main_Lam"]
    C = suppress_steps_numpy((H - op.a["mean"]) @ op.a["main_B_read"], Lam, float(op.a["main_eps0"]))
    checks["suppress_within_budget"] = bool(np.all(np.linalg.norm(C, axis=1) <= float(op.a["main_eps0"]) * (1 + 1e-9)))
    passed = (checks["cos_main_edit_vs_planted"] > 0.9 and checks["cos_euclid_edit_vs_planted"] < 0.6 and checks["signed_main_edit_on_planted"] > 0
              and checks["cos_random_edit_vs_planted"] < 0.3 and checks["sham_identical"] and checks["strengthen_matches_closed_form"] and checks["suppress_reduces_causal_energy"]
              and checks["suppress_within_budget"] and checks["D_action_main"] <= checks["tau2"] * (1 + 1e-6) and checks["mahal_incr_main"] <= checks["mahal_incr_bound"] * (1 + 1e-6)
              and checks["loso"]["main"]["mean"] > checks["loso"]["euclid"]["mean"])
    checks["numpy_passed"] = bool(passed)
    # torch hook check on a fake predictor
    try:
        import torch

        from predictor_hooks import Site
        from steered_planner_ranking import PredictorSteerer

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        torch.manual_seed(seed)
        d = op.d
        fake = _FakeWM(d, 2, 3)
        z = torch.randn(1, 1, 1, 2, 2, d)
        acts = torch.randn(3, 1, 3)
        base = fake.unroll(z, act_suffix=acts)
        tokens = np.array([1, 2])
        for beta, mode in ((0.0, "strengthen"), (1.0, "strengthen"), (1.0, "suppress")):
            fn = op.torch_fn(mode, beta, "main", torch.device("cpu"))
            st = PredictorSteerer(fake.model.predictor, {"L01.attn_out": [(0, tokens, fn)]}, 4, Site)
            with st, torch.inference_mode():
                out_ = fake.unroll(z, act_suffix=acts)
            if beta == 0.0:
                checks["torch_sham_bit_identical"] = bool(torch.equal(out_, base)) and bool(st.applied)
            else:
                checks[f"torch_{mode}_changes_output"] = bool(not torch.equal(out_, base)) and bool(st.applied)
        # per-token torch vs numpy agreement for suppress
        h = torch.randn(1, 3, d) + 2.0 * torch.as_tensor(u, dtype=torch.float32)
        fn = op.torch_fn("suppress", 1.0, "main", torch.device("cpu"))
        checks["torch_suppress_matches_numpy"] = bool(np.allclose(fn(h)[0].numpy(), op.apply_numpy(h[0].numpy(), "suppress", 1.0), atol=1e-3))
        checks["torch_passed"] = bool(checks["torch_sham_bit_identical"] and checks["torch_strengthen_changes_output"] and checks["torch_suppress_changes_output"] and checks["torch_suppress_matches_numpy"])
    except ImportError as exc:  # torch not installed: numpy part only
        checks["torch_passed"] = None
        checks["torch_skipped"] = str(exc)
    checks["passed"] = bool(checks["numpy_passed"] and checks["torch_passed"] is not False)
    checks["runtime_s"] = time.time() - t0
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "self_test.json").write_text(json.dumps(finite(checks), indent=1) + "\n")
    print(json.dumps(finite({k: v for k, v in checks.items() if k != "loso"}), indent=1))
    print("SELF_TEST_PASSED" if checks["passed"] else "SELF_TEST_FAILED")
    return checks


class _FakeWM:
    """Minimal AdaLN-shaped predictor (same module names as ``predictor_hooks`` expects) for the hook check."""

    def __init__(self, d: int, grid: int, depth: int) -> None:
        import torch
        import torch.nn as nn

        N = grid * grid

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1, self.attn, self.norm2 = nn.LayerNorm(d), nn.Linear(d, d), nn.LayerNorm(d)
                self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

            def forward(self, x, z, T=None):
                sh, sc, g1, sh2, sc2, g2 = self.adaLN_modulation(z).repeat_interleave(N, dim=1).chunk(6, dim=2)
                x = x + self.attn(self.norm1(x) * (1 + sc) + sh) * g1
                return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)

        class Pred(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor_embed, self.action_encoder = nn.Linear(d, d), nn.Linear(3, d)
                self.predictor_blocks = nn.ModuleList([Block() for _ in range(depth)])
                self.predictor_norm, self.predictor_proj = nn.LayerNorm(d), nn.Linear(d, d)

            def forward(self, x, actions):
                x = self.predictor_embed(x).flatten(2, 4)
                B, T = x.shape[:2]
                z = self.action_encoder(actions)
                x = x.flatten(1, 2)
                for blk in self.predictor_blocks:
                    x = blk(x, z, T=T)
                return self.predictor_proj(self.predictor_norm(x)).view(B, T, N, d)

        class VWM(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = Pred()

        self.model = VWM().eval()
        self.grid_size, self.ctxt_window, self._d, self._N = grid, 2, d, N

    def unroll(self, z_ctxt, act_suffix=None):
        import torch

        T, B, _ = act_suffix.shape
        vid = z_ctxt.expand(B, *z_ctxt.shape[1:])
        acts = act_suffix.permute(1, 0, 2)
        for h in range(T):
            a = acts[:, : h + 1][:, -self.ctxt_window:]
            pred = self.model.predictor(vid[:, -self.ctxt_window:], a)
            vid = torch.cat([vid, pred[:, -1:].view(B, 1, 1, self.grid_size, self.grid_size, self._d)], dim=1)
        return vid.permute(1, 0, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit", help="fit per-site operators from a localization dump (discovery scenes only)")
    f.add_argument("--dump", type=Path, required=True)
    f.add_argument("--stimulus", type=Path, required=True)
    f.add_argument("--discovery-seeds", type=Path, default=None)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--sites", nargs="*", default=None)
    f.add_argument("--steps", type=int, nargs="*", default=[0])
    f.add_argument("--groups", nargs="*", default=["hazard_corridor"])
    f.add_argument("--solid-level", type=int, choices=[1, OBJECT_HAZARD], default=1)
    f.add_argument("--domain", default="driving")
    f.add_argument("--rank", type=int, default=4, help="edit subspace rank (frozen grid 1/2/4)")
    f.add_argument("--k-action", type=int, default=4)
    f.add_argument("--lam-rel", type=float, default=0.1)
    f.add_argument("--tau-action-rel", type=float, default=0.1)
    f.add_argument("--mahal-incr", type=float, default=0.25)
    f.add_argument("--no-token-nuisance", dest="token_nuisance", action="store_false")
    f.add_argument("--max-token-rows", type=int, default=4000)
    f.add_argument("--no-manifest-factors", dest="manifest_factors", action="store_false")
    f.add_argument("--no-loso", action="store_true")
    f.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pullback", help="JVP pullback check of fitted operators (needs the model; box only)")
    p.add_argument("--operator-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-name", default="jepa_wm_driving")
    p.add_argument("--artifacts", type=Path, nargs="+", required=True)
    p.add_argument("--seeds-file", type=Path, default=None)
    p.add_argument("--domain", default="driving")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-scenes", type=int, default=8)
    p.add_argument("--tau-pullback", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> Any:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(Path("artifacts/causal_metric_self_test") if argv is None else None, args.seed)
    if args.cmd == "fit":
        return fit_from_dump(args)
    if args.cmd == "pullback":
        return pullback_check(args)
    ap.print_help()
    return None


if __name__ == "__main__":
    main()
