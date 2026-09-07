"""CPU tests for the higher-order arm: equal-covariance / different-skew distributions are detected only after
moment matching; a planted rotating subspace transfers only after Procrustes transport; kernel HSIC catches a
nonlinear dependence that linear CKA misses; a spatially distributed interaction needs the two-sided operator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))
sys.path.insert(0, str(ROOT / "tests"))

import geometry_higher_order as gh  # noqa: E402
from test_geometry_localize import D, build_dump, build_stimulus  # noqa: E402


def test_sliced_w2_skew_detected_only_after_moment_matching():
    rng = np.random.default_rng(0)
    n, d, S = 200, 16, 10  # per-token row counts on the real dump are 100-800 per hazard level
    scene = np.repeat(np.arange(S), n // S)
    # both groups share the mean and covariance (3 dominant axes, small isotropic rest); the H1 rows are skewed
    # (exponential) on the dominant axes.  Random projections of many iid skewed coordinates go Gaussian by the
    # CLT, so a realistic higher-order signature must live on a few dominant directions -- which is exactly what
    # the covariance-eigenvector projection family targets.
    sd = np.concatenate([np.ones(3), np.full(d - 3, 0.1)])
    g = rng.normal(size=(n, d)) * sd
    e = np.concatenate([rng.exponential(size=(n, 3)) - 1.0, rng.normal(size=(n, d - 3)) * 0.1], axis=1)
    rows = np.concatenate([g, e])
    hz = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    sc = np.concatenate([scene, scene])
    proj = gh.projection_set(rows, n_random=64, seed=0, n_eig=4)
    flips = np.random.default_rng(1).choice([-1.0, 1.0], size=(199, S))
    res = gh.cramer_wold_arm(rows, hz, sc, proj, flips, top_k=8)
    assert res["eig|after"]["topk_mean"]["p_raw"] < 0.05
    assert res["random|after"]["topk_mean"]["p_raw"] < 0.05
    # control: two Gaussian samples with the same covariance -> nothing after matching
    rows2 = np.concatenate([rng.normal(size=(n, d)) * sd, rng.normal(size=(n, d)) * sd])
    res2 = gh.cramer_wold_arm(rows2, hz, sc, proj, flips, top_k=8)
    assert res2["eig|after"]["topk_mean"]["p_raw"] > 0.05
    assert res2["random|after"]["topk_mean"]["p_raw"] > 0.05
    # a pure mean shift is visible before matching and gone after
    rows3 = np.concatenate([rng.normal(size=(n, d)) * sd, rng.normal(size=(n, d)) * sd + 1.5 * sd])
    res3 = gh.cramer_wold_arm(rows3, hz, sc, proj, flips, top_k=8)
    assert res3["random|before"]["topk_mean"]["p_raw"] < 0.05
    assert res3["random|after"]["topk_mean"]["p_raw"] > 0.05
    assert res3["loso_disc|before"]["topk_mean"]["p_raw"] < 0.05


def test_hsic_kernel_detects_nonlinear_dependence_linear_cka_does_not():
    rng = np.random.default_rng(2)
    n, d = 120, 8
    X = rng.normal(size=(n, d))
    y = X[:, 0] ** 2 + 0.1 * rng.normal(size=n)  # symmetric: zero linear correlation
    scene = np.repeat(np.arange(30), 4)
    res = gh.hsic_arm(X, {"y": (y, "continuous")}, scene, n_perm=199, seed=0)
    assert res["y"]["p_nhsic"] < 0.05
    assert res["y"]["p_cka"] > 0.05
    ylin = X[:, 0] + 0.1 * rng.normal(size=n)
    res2 = gh.hsic_arm(X, {"y": (ylin, "continuous")}, scene, n_perm=199, seed=0)
    assert res2["y"]["p_cka"] < 0.05 and res2["y"]["p_nhsic"] < 0.05


def test_rotating_subspace_transfers_only_after_procrustes():
    rng = np.random.default_rng(3)
    d, k, S = 48, 4, 30
    scene = np.repeat(np.arange(S), 4)
    hz = np.tile([0, 0, 1, 1], S)
    ac = np.tile([0, 1, 0, 1], S)
    y = ((2 * hz - 1) * (2 * ac - 1) > 0).astype(float)
    Ua = np.linalg.qr(rng.normal(size=(d, k)))[0]
    # target basis: the same subspace family rotated to a different set of directions (U_b = R U_a) so that raw
    # projection on U_a of target activations sees nothing; Procrustes on the bases recovers the alignment
    R = np.linalg.qr(rng.normal(size=(d, d)))[0]
    Ub = R @ Ua
    coeff = rng.normal(size=(S, k)) * np.array([3.0, 2.5, 2.0, 1.5])
    w = np.array([1.0, -0.5, 0.8, -0.3])
    Da = coeff @ Ua.T + rng.normal(size=(S, d)) * 0.05
    Db = coeff @ Ub.T + rng.normal(size=(S, d)) * 0.05
    # cells: interaction contrast times the scene's interaction vector, readout label = sign of w . coeff (per cell contrast)
    lab = (coeff @ w > 0).astype(float)
    Xa = np.stack([(2 * hz[i] - 1) * (2 * ac[i] - 1) / 4 * Da[scene[i]] * (2 * lab[scene[i]] - 1) for i in range(4 * S)]) + rng.normal(size=(4 * S, d)) * 0.02
    Xb = np.stack([(2 * hz[i] - 1) * (2 * ac[i] - 1) / 4 * Db[scene[i]] * (2 * lab[scene[i]] - 1) for i in range(4 * S)]) + rng.normal(size=(4 * S, d)) * 0.02
    tr = gh.transfer_test(Xa, Xb, Da, Db, scene, np.arange(S), y, k, lam=1e-2)
    assert tr["status"] == "ok"
    assert tr["source"] > 0.9
    assert tr["transported"] > 0.85, tr
    assert tr["raw"] < 0.7, tr
    ang = gh.principal_angles(Ua, Ub)
    assert np.degrees(ang).min() > 30
    assert gh.grassmann_distance(Ua, Ua) < 1e-6


def test_two_sided_operator_detects_spatial_distribution():
    rng = np.random.default_rng(4)
    S, T, d = 10, 256, 32
    V = np.linalg.qr(rng.normal(size=(d, 2)))[0]
    W = np.linalg.qr(rng.normal(size=(T, 2)))[0]
    # distributed: rank-2 token pattern x rank-2 hidden pattern
    dist = np.stack([W @ (rng.normal(size=(2, 2)) * 5) @ V.T + rng.normal(size=(T, d)) * 0.02 for _ in range(S)])
    # single token: all energy at one grid cell
    single = np.zeros((S, T, d))
    for s in range(S):
        single[s, 57] = rng.normal(size=2) @ V.T * 5
        single[s] += rng.normal(size=(T, d)) * 0.02
    rd = gh.two_sided_check(dist, 4, 4)
    rs = gh.two_sided_check(single, 4, 4)
    assert rd["spatially_distributed"] and rd["token_participation_ratio"] > 2.0
    assert not rs["spatially_distributed"] and rs["token_participation_ratio"] < 1.5
    assert rd["two_sided"]["explained"] > 0.8


def test_end_to_end_on_fake_dump(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, 12)
    rng = np.random.default_rng(5)
    X = np.zeros((len(cells), D))
    u = rng.normal(size=D)
    for c in cells:
        s = int(c["seed"]) - 300
        h, a = c["hazard"], c["candidate_action"]
        # H1 action effect skewed, H0 action effect Gaussian, same second moments: a higher-order hazard signature
        X[c["row"]] = rng.normal(size=D) * 0.1 + a * (rng.exponential(size=D) - 1.0 if h else rng.normal(size=D)) + (2 * h - 1) * (2 * a - 1) / 4 * u
    dump = build_dump(tmp_path, stim, cells, X, sites=("L07.attn_out", "L08.attn_out"))
    seeds = tmp_path / "disc.txt"
    seeds.write_text("\n".join(str(300 + s) for s in range(12)))
    rep = gh.main(["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "out"), "--discovery-seeds", str(seeds),
                   "--steps", "0", "1", "--groups", "egg", "--n-perm", "99", "--n-perm-hsic", "49", "--n-random", "32"])
    assert (tmp_path / "out" / "higher_order_map.json").exists()
    e = [x for x in rep["entries"] if x["site_id"] == "L07.attn_out" and x["step"] == 0][0]
    assert e["status"] == "ok"
    cw = e["cramer_wold"]["token"]
    for fam in ("eig|before", "eig|after", "random|before", "random|after", "loso_disc|before", "loso_disc|after"):
        assert fam in cw and "p_maxstat" in cw[fam]["topk_mean"]
    assert "interaction" in e["hsic"]["pooled_interaction_residual"]
    assert e["two_sided"]["status"] == "ok"
    assert "participation_ratio" in e["descriptives"]["pooled_interaction"]
    assert "L07.attn_out->L08.attn_out|s0|egg" in rep["tracking"]["adjacent_layers"]
    assert "L07.attn_out|egg|s0->s1" in rep["tracking"]["adjacent_steps"]
    assert "step0|egg" in rep["ranked_tables"] and rep["ranked_tables"]["step0|egg"][0]["site_id"] in ("L07.attn_out", "L08.attn_out")
    assert rep["focus_L07_attn_out_step0"][0]["site_id"] == "L07.attn_out"
