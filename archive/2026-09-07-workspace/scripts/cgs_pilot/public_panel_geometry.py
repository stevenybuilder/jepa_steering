#!/usr/bin/env python3
"""Panel P — regime geometry and conceptor transport (research-thread extensions).

Implements, on the Panel P captures (`public_panel_capture.py`), the ideas from
`vla_world_model_research_thread_codex (1).md` that are testable with what we already record.
All of it is CPU-only and read-only; it produces operators that `public_panel_coast.py steer`
can apply, so every claim still ends at the released evaluator's native success flag.

§56 `regimes` — the minimal two-regime experiment, run BEFORE anything heavier.
    Regime axis: the CEM optimiser's own progress. Rows are stored in call order, so contiguous
    blocks of planner rows correspond to early (exploratory) vs late (converged-elite) iterations.
    Per regime r we fit (mu_r, Sigma_r) and the contrastive conceptors C_s^r, C_f^r,
    C_steer^r = C_s^r AND NOT C_f^r, and report:
      * center distance ||mu_A - mu_B|| (and normalised by the pooled scale),
      * principal angles + Grassmann distance between the top-k covariance eigenspaces,
      * linear CKA between regimes (basis-invariant relational comparison),
      * conceptor overlap tr(AB)/sqrt(tr(A^2)tr(B^2)).
    Verdict `regimes_differ` is a scene-permutation test on the regime label.

§34 `regimes` also answers "does one global conceptor blur local ones?":
      C_global vs the mixture sum_r P(r) C_r  (Frobenius residual, overlaps).

§42/§43 `transport` — Procrustes transport of a conceptor between two models on the SAME task
    (their captures share the seed, hence the same goals, so episodes are matched anchors):
      Q* = argmin_Q ||Q X_A - X_B||_F  s.t. Q^T Q = I      (orthogonal, preserves norms/angles)
      C_hat_B = Q C_A Q^T ,  Sigma_hat_B = Q Sigma_A Q^T ,  T(h) = mu_B + Q (h - mu_A)
    This is the cross-model test COAST does not do: it reuses a conceptor across tasks directly,
    never through a fitted change of basis. Only defined when both models share the site width.

Usage
    python public_panel_geometry.py regimes   --capture <fit dir> --out <dir> [--n-regimes 2] [--site S]
    python public_panel_geometry.py transport --capture-a <dir> --capture-b <dir> \
        --operators-a <coast dir>/operators --out <dir> [--site S]
    python public_panel_geometry.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_panel_coast import conceptor, c_and, c_not, quota, overlap  # noqa: E402


# ------------------------------------------------------------------ geometry helpers
def top_eigspace(S: np.ndarray, k: int) -> np.ndarray:
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V[:, np.argsort(w)[::-1][:k]]


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def grassmann(U: np.ndarray, V: np.ndarray) -> float:
    return float(np.sqrt((principal_angles(U, V) ** 2).sum()))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Basis-invariant relational similarity of two representations of the SAME inputs (rows must be
    matched pairs). Not applicable to two different samples — regimes are compared with principal
    angles / Grassmann / KL instead; CKA is used only for the matched cross-model anchors."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den else float("nan")


def gauss_stats(X: np.ndarray, eps: float = 1e-6) -> dict:
    """Distributional spread of an activation cloud (research thread §38, "activation density"
    P(h | z=r), which the thread separates from attention allocation P(j|i) and belief P(z|h)).
    Reports the Gaussian differential entropy 0.5 log det(Sigma) (up to the constant), the
    participation ratio (effective dimensionality) and the trace, so two subspaces can be compared
    on SPREAD, not only on orientation."""
    Xc = X - X.mean(0, keepdims=True)
    S = (Xc.T @ Xc) / max(len(Xc), 1)
    w = np.clip(np.linalg.eigvalsh((S + S.T) / 2), 0, None)
    wp = w[w > eps * max(w.max(), 1e-12)]
    ent = float(0.5 * np.log(wp + 1e-30).sum()) if len(wp) else float("nan")
    pr = float((w.sum() ** 2) / (np.square(w).sum() + 1e-30))
    return {"gaussian_entropy_logdet_half": ent, "participation_ratio": pr,
            "trace": float(w.sum()), "rank_used": int(len(wp))}


def gauss_kl(Xp: np.ndarray, Xq: np.ndarray, ridge: float = 1e-3) -> float:
    """KL(N_p || N_q) between Gaussian fits of two clouds — a single number for
    "these two subspaces do not carry the same distribution" (§38)."""
    def ms(X):
        m = X.mean(0); Xc = X - m
        S = (Xc.T @ Xc) / max(len(Xc), 1)
        return m, S + ridge * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    mp, Sp = ms(Xp); mq, Sq = ms(Xq)
    d = Sp.shape[0]
    Sq_i = np.linalg.pinv(Sq)
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    dm = (mq - mp)
    return float(0.5 * (np.trace(Sq_i @ Sp) + dm @ Sq_i @ dm - d + ldq - ldp))


def load_rows(capture: Path, pool: str = "planner"):
    """-> {site: (X [N,D], y [N], g [N] episode, r [N] within-episode row order)}"""
    idx = json.loads((capture / "activations" / "index.json").read_text())
    per: dict[str, dict[str, list]] = {}
    for ei, row in enumerate(idx["episodes"]):
        z = np.load(capture / "activations" / row["file"])
        y = int(row.get("success", 0))
        for k in z.files:
            if not k.endswith(f"__{pool}"):
                continue
            site = k.split("__")[0]
            v = z[k]
            d = per.setdefault(site, {"X": [], "y": [], "g": [], "r": []})
            d["X"].append(v)
            d["y"].append(np.full(len(v), y))
            d["g"].append(np.full(len(v), ei))
            d["r"].append(np.linspace(0.0, 1.0, len(v)))     # position within the CEM's call order
    return {s: (np.concatenate(d["X"]), np.concatenate(d["y"]),
                np.concatenate(d["g"]), np.concatenate(d["r"])) for s, d in per.items()}, idx


# ------------------------------------------------------------------ §56 / §34
def cmd_regimes(args) -> int:
    cap, out = Path(args.capture), Path(args.out)
    (out / "operators").mkdir(parents=True, exist_ok=True)
    per, idx = load_rows(cap, args.pool)
    if not per:
        print(json.dumps({"error": "no rows"})); return 1
    sites = [args.site] if args.site else sorted(per)
    R = int(args.n_regimes)
    rng = np.random.default_rng(args.seed)
    report = {"capture": str(cap), "n_regimes": R, "pool": args.pool,
              "regime_axis": "CEM call order within the episode's single plan call (early=exploration, late=converged elites)",
              "sites": []}

    for site in sites:
        X, y, g, r = per[site]
        edges = np.linspace(0, 1, R + 1)
        lab = np.clip(np.digitize(r, edges[1:-1]), 0, R - 1)
        row = {"site": site, "dim": int(X.shape[1]), "n_rows": int(len(X)),
               "n_per_regime": [int((lab == i).sum()) for i in range(R)]}
        # §38: activation-density comparison, regimes and outcome classes
        row["density_by_regime"] = [gauss_stats(X[lab == i]) for i in range(R)]
        if (y == 1).sum() > 2 and (y == 0).sum() > 2:
            row["density_success"] = gauss_stats(X[y == 1])
            row["density_failure"] = gauss_stats(X[y == 0])
            row["kl_success_failure"] = gauss_kl(X[y == 1], X[y == 0])
            row["kl_failure_success"] = gauss_kl(X[y == 0], X[y == 1])

        mus, sigs, csteer = [], [], []
        for i in range(R):
            m = lab == i
            Xi = X[m]
            mus.append(Xi.mean(0))
            Xc = Xi - Xi.mean(0, keepdims=True)
            sigs.append((Xc.T @ Xc) / max(len(Xc), 1))
            ys, yf = Xi[y[m] == 1], Xi[y[m] == 0]
            if len(ys) > 2 and len(yf) > 2:
                csteer.append(c_and(conceptor(ys, args.alpha), c_not(conceptor(yf, args.alpha))))
            else:
                csteer.append(None)

        # pairwise regime comparison (§56 Q1)
        k = int(args.k_eig)
        pairs = []
        for i in range(R):
            for j in range(i + 1, R):
                Ui, Uj = top_eigspace(sigs[i], k), top_eigspace(sigs[j], k)
                scale = float(np.sqrt(np.trace(sigs[i]) + np.trace(sigs[j])) + 1e-12)
                p = {"regimes": [i, j],
                     "center_distance": float(np.linalg.norm(mus[i] - mus[j])),
                     "center_distance_normalised": float(np.linalg.norm(mus[i] - mus[j]) / scale),
                     "grassmann_topk": grassmann(Ui, Uj),
                     "principal_angles_deg": [round(float(a * 180 / np.pi), 2) for a in principal_angles(Ui, Uj)][:5],
                     "kl_regime_i_j": gauss_kl(X[lab == i], X[lab == j]),
                     "entropy_gap": (row["density_by_regime"][i]["gaussian_entropy_logdet_half"]
                                     - row["density_by_regime"][j]["gaussian_entropy_logdet_half"])}
                if csteer[i] is not None and csteer[j] is not None:
                    p["conceptor_overlap"] = overlap(csteer[i], csteer[j])   # §56 Q2: C_s^A vs C_s^B
                pairs.append(p)
        row["pairs"] = pairs

        # §34: global vs mixture of local conceptors
        ys_all, yf_all = X[y == 1], X[y == 0]
        if len(ys_all) > 2 and len(yf_all) > 2:
            Cg = c_and(conceptor(ys_all, args.alpha), c_not(conceptor(yf_all, args.alpha)))
            have = [i for i in range(R) if csteer[i] is not None]
            if have:
                P = np.array([(lab == i).mean() for i in have]); P = P / P.sum()
                mix = sum(p * csteer[i] for p, i in zip(P, have))
                row["global_vs_mixture"] = {
                    "quota_global": quota(Cg),
                    "quota_local": [quota(csteer[i]) for i in have],
                    "overlap_global_local": [overlap(Cg, csteer[i]) for i in have],
                    "frobenius_residual_global_minus_mixture": float(np.linalg.norm(Cg - mix, "fro")),
                    "frobenius_norm_global": float(np.linalg.norm(Cg, "fro")),
                }
            for i in have:
                np.savez_compressed(
                    out / "operators" / f"{site}__regime{i}.npz",
                    C_steer=csteer[i].astype(np.float32), mean=mus[i].astype(np.float32),
                    mean_success=X[(lab == i) & (y == 1)].mean(0).astype(np.float32),
                    mean_failure=X[(lab == i) & (y == 0)].mean(0).astype(np.float32),
                    C_success=conceptor(X[(lab == i) & (y == 1)], args.alpha).astype(np.float32),
                    C_failure=conceptor(X[(lab == i) & (y == 0)], args.alpha).astype(np.float32),
                    rank_one=np.zeros_like(csteer[i], dtype=np.float32),
                    random_matched=csteer[i].astype(np.float32), alpha=np.float32(args.alpha))

        # permutation test: shuffle the regime label WITHIN each episode
        obs = np.mean([p["grassmann_topk"] for p in pairs]) if pairs else np.nan
        null = []
        for _ in range(int(args.n_perm)):
            lp = lab.copy()
            for e in np.unique(g):
                m = g == e
                lp[m] = rng.permutation(lp[m])
            gs = []
            for i in range(R):
                for j in range(i + 1, R):
                    Si = np.cov((X[lp == i] - X[lp == i].mean(0)).T) if (lp == i).sum() > 2 else None
                    Sj = np.cov((X[lp == j] - X[lp == j].mean(0)).T) if (lp == j).sum() > 2 else None
                    if Si is None or Sj is None:
                        continue
                    gs.append(grassmann(top_eigspace(Si, k), top_eigspace(Sj, k)))
            if gs:
                null.append(float(np.mean(gs)))
        if null:
            row["grassmann_obs"] = float(obs)
            row["grassmann_null_mean"] = float(np.mean(null))
            row["p_regimes_differ"] = float((np.array(null) >= obs).mean())
            row["regimes_differ"] = bool(row["p_regimes_differ"] < 0.05)
        report["sites"].append(row)

    (out / "regimes.json").write_text(json.dumps(report, indent=2))
    best = sorted([r for r in report["sites"] if "p_regimes_differ" in r],
                  key=lambda r: r["p_regimes_differ"])[:3]
    print(json.dumps({"out": str(out), "n_sites": len(report["sites"]),
                      "most_regime_dependent": [{k: b[k] for k in ("site", "grassmann_obs", "grassmann_null_mean", "p_regimes_differ")} for b in best]}))
    return 0


# ------------------------------------------------------------------ §42 / §43
def cmd_transport(args) -> int:
    ca, cb, out = Path(args.capture_a), Path(args.capture_b), Path(args.out)
    (out / "operators").mkdir(parents=True, exist_ok=True)
    pa, ia = load_rows(ca, args.pool)
    pb, ib = load_rows(cb, args.pool)
    ops_a = Path(args.operators_a)
    sites = [args.site] if args.site else sorted(set(pa) & set(pb))
    rows = []
    for site in sites:
        Xa, ya, ga, _ = pa[site]
        Xb, yb, gb, _ = pb[site]
        if Xa.shape[1] != Xb.shape[1]:
            rows.append({"site": site, "skipped": f"width {Xa.shape[1]} vs {Xb.shape[1]}"}); continue
        # matched anchors: the same episode index = the same goal/seed under both models
        eps = sorted(set(ga.tolist()) & set(gb.tolist()))
        if len(eps) < 4:
            rows.append({"site": site, "skipped": "fewer than 4 matched episodes"}); continue
        A = np.stack([Xa[ga == e].mean(0) for e in eps])
        B = np.stack([Xb[gb == e].mean(0) for e in eps])
        mu_a, mu_b = A.mean(0), B.mean(0)
        # orthogonal Procrustes on centred anchors: Q* = argmin ||Q A^T - B^T||_F, Q^T Q = I
        U, _, Vt = np.linalg.svd((A - mu_a).T @ (B - mu_b))
        Q = U @ Vt
        resid = float(np.linalg.norm((A - mu_a) @ Q - (B - mu_b), "fro") / (np.linalg.norm(B - mu_b, "fro") + 1e-12))
        # held-out (leave-one-episode-out) transport error, so the map is not scored on its own fit
        loo = []
        for h, e in enumerate(eps):
            keep = [i for i in range(len(eps)) if i != h]
            Uh, _, Vth = np.linalg.svd((A[keep] - A[keep].mean(0)).T @ (B[keep] - B[keep].mean(0)))
            Qh = Uh @ Vth
            pred = B[keep].mean(0) + (A[h] - A[keep].mean(0)) @ Qh
            loo.append(float(np.linalg.norm(pred - B[h]) / (np.linalg.norm(B[h] - B[keep].mean(0)) + 1e-12)))
        rnd = np.linalg.qr(np.random.default_rng(0).standard_normal((Xa.shape[1], Xa.shape[1])))[0]
        rnd_resid = float(np.linalg.norm((A - mu_a) @ rnd - (B - mu_b), "fro") / (np.linalg.norm(B - mu_b, "fro") + 1e-12))
        rec = {"site": site, "n_anchor_episodes": len(eps),
               "cka_matched_anchors": linear_cka(A, B),
               "procrustes_relative_residual": resid,
               "loo_relative_error_median": float(np.median(loo)), "random_orthogonal_residual": rnd_resid,
               "beats_random": bool(resid < rnd_resid)}
        opf = ops_a / f"{site}.npz"
        if opf.exists():
            z = np.load(opf)
            Ca = z["C_steer"]
            Cb_hat = Q.T @ Ca @ Q            # C_hat_B = Q C_A Q^T with our row-vector convention
            np.savez_compressed(
                out / "operators" / f"{site}.npz",
                C_steer=Cb_hat.astype(np.float32),
                C_success=(Q.T @ z["C_success"] @ Q).astype(np.float32),
                C_failure=(Q.T @ z["C_failure"] @ Q).astype(np.float32),
                mean=mu_b.astype(np.float32),
                mean_success=(mu_b + (z["mean_success"] - mu_a) @ Q).astype(np.float32),
                mean_failure=(mu_b + (z["mean_failure"] - mu_a) @ Q).astype(np.float32),
                rank_one=np.zeros_like(Cb_hat, dtype=np.float32),
                random_matched=(rnd.T @ Ca @ rnd).astype(np.float32),   # same conceptor, random basis
                Q=Q.astype(np.float32), mu_a=mu_a.astype(np.float32), alpha=z["alpha"])
            rec["operator_written"] = True
            rec["overlap_transported_vs_source"] = overlap(Cb_hat, Ca)
        rows.append(rec)
    rep = {"capture_a": str(ca), "capture_b": str(cb), "operators_a": str(ops_a), "sites": rows}
    (out / "transport.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"out": str(out), "n_sites": len(rows),
                      "beats_random": sum(1 for r in rows if r.get("beats_random"))}))
    return 0


# ------------------------------------------------------------------ self-test
def self_test() -> int:
    rng = np.random.default_rng(0)
    D, k = 48, 6
    # two regimes with genuinely different covariance eigenspaces
    Ba = np.linalg.qr(rng.standard_normal((D, D)))[0][:, :k]
    Bb = np.linalg.qr(rng.standard_normal((D, D)))[0][:, :k]
    Xa = rng.standard_normal((400, k)) @ Ba.T + 0.05 * rng.standard_normal((400, D))
    Xb = rng.standard_normal((400, k)) @ Bb.T + 0.05 * rng.standard_normal((400, D))
    g_diff = grassmann(top_eigspace(np.cov(Xa.T), k), top_eigspace(np.cov(Xb.T), k))
    Xa2 = rng.standard_normal((400, k)) @ Ba.T + 0.05 * rng.standard_normal((400, D))
    g_same = grassmann(top_eigspace(np.cov(Xa.T), k), top_eigspace(np.cov(Xa2.T), k))
    # CKA is only meaningful on matched rows: same inputs, two representations
    R1 = rng.standard_normal((200, k))
    P1, P2 = R1 @ Ba.T, R1 @ Ba.T @ np.linalg.qr(rng.standard_normal((D, D)))[0]   # same inputs, rotated view
    cka_same = linear_cka(P1, P2)
    cka_diff = linear_cka(P1, rng.standard_normal((200, D)))
    # transport: B is a rotated copy of A, so Procrustes must beat a random orthogonal map
    Qtrue = np.linalg.qr(rng.standard_normal((D, D)))[0]
    A = rng.standard_normal((30, D)); Bm = A @ Qtrue + 0.01 * rng.standard_normal((30, D))
    U, _, Vt = np.linalg.svd((A - A.mean(0)).T @ (Bm - Bm.mean(0))); Q = U @ Vt
    res = np.linalg.norm((A - A.mean(0)) @ Q - (Bm - Bm.mean(0)), "fro") / np.linalg.norm(Bm - Bm.mean(0), "fro")
    rnd = np.linalg.qr(rng.standard_normal((D, D)))[0]
    res_rnd = np.linalg.norm((A - A.mean(0)) @ rnd - (Bm - Bm.mean(0)), "fro") / np.linalg.norm(Bm - Bm.mean(0), "fro")
    ok = (g_diff > 3 * g_same) and (cka_same > 0.9 > cka_diff) and (res < 0.1 < res_rnd)
    # (CKA on matched rows: a rotation preserves the Gram matrix, so cka_same must be ~1)
    print(json.dumps({"grassmann_different": round(g_diff, 3), "grassmann_same": round(g_same, 3),
                      "cka_different": round(cka_diff, 3), "cka_same": round(cka_same, 3),
                      "procrustes_residual": round(float(res), 4), "random_residual": round(float(res_rnd), 4),
                      "SELF_TEST": "PASS" if ok else "FAIL"}))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("regimes")
    r.add_argument("--capture", required=True); r.add_argument("--out", required=True)
    r.add_argument("--pool", default="planner"); r.add_argument("--site", default=None)
    r.add_argument("--n-regimes", type=int, default=2); r.add_argument("--alpha", type=float, default=5.0)
    r.add_argument("--k-eig", type=int, default=8); r.add_argument("--n-perm", type=int, default=50)
    r.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("transport")
    t.add_argument("--capture-a", required=True); t.add_argument("--capture-b", required=True)
    t.add_argument("--operators-a", required=True); t.add_argument("--out", required=True)
    t.add_argument("--pool", default="planner"); t.add_argument("--site", default=None)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.cmd == "regimes":
        return cmd_regimes(args)
    if args.cmd == "transport":
        return cmd_transport(args)
    ap.print_help(); return 2


if __name__ == "__main__":
    raise SystemExit(main())
