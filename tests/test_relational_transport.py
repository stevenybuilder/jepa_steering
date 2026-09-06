"""CPU tests for relational_transport.py: field definitions are the registered DiD contrasts; the Gram-space transport
scorer equals the explicit-vector computation (observed, permutation, covariance-matched and rotated nulls, isotropic
null distribution); residual cosine of the field-mean baseline is 0 and the copy-delta donor is another scene; the
vectorised bootstrap reproduces stats_utils.cluster_bootstrap_mean exactly; tie-aware Spearman matches a hand
computation; the slim cache round-trips a full latent_cache.py npz; the planted synthetic self-test passes and the
null variant does not transport; the JSON/Markdown outputs carry interpretation_scope."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import relational_transport as rt  # noqa: E402
from stats_utils import cluster_bootstrap_mean  # noqa: E402


def _cells(y):
    return {(h, a): i for i, (h, a) in enumerate([(h, a) for h in (0, 1, 2, 3) for a in (0, 1)])}


def test_field_definitions_are_the_registered_contrasts():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(8, 16, 16, 4))
    cc = _cells(y)
    tok = np.array([5, 17, 40])
    P = {k: y[i].reshape(256, 4)[tok].ravel() for k, i in cc.items()}
    ped = P[(1, 1)] - P[(1, 0)] - P[(0, 1)] + P[(0, 0)]
    cone = P[(3, 1)] - P[(3, 0)] - P[(0, 1)] + P[(0, 0)]
    null = P[(2, 1)] - P[(2, 0)] - P[(0, 1)] + P[(0, 0)]
    fv = lambda name: rt.field_vector(y, cc, tok, rt.FIELD_DEFS[name], pool=False)  # noqa: E731
    assert np.allclose(fv("ped_did"), ped)
    assert np.allclose(fv("cone_did"), cone)
    assert np.allclose(fv("ped_did_relational"), ped - null)
    assert np.allclose(fv("cone_did_relational"), cone - null)
    assert np.allclose(fv("action"), P[(0, 1)] - P[(0, 0)])
    # pooled representation (default) = mean over the tokens -> [D]
    assert np.allclose(rt.field_vector(y, cc, tok, rt.FIELD_DEFS["ped_did"]), ped.reshape(3, 4).mean(0))
    # a scene without the level-3 cells has no cone field
    cc2 = {k: v for k, v in cc.items() if k[0] != 3}
    assert rt.field_vector(y, cc2, tok, rt.FIELD_DEFS["cone_did"]) is None


def _explicit_scores(X, T):
    """Explicit-vector raw / residual cosine of prediction rows X against T with the LOSO mean of T."""

    G = rt.loso_mean(T)
    return {"raw": rt.row_cos(X, T), "residual": rt.row_cos(X - G, T - G), "norm_pred": np.linalg.norm(X, axis=1)}


def test_gram_scorer_matches_explicit_vectors():
    rng = np.random.default_rng(1)
    n, d = 9, 40
    S = rng.normal(size=(n, d)) + 2.0
    T = rng.normal(size=(n, d)) + 1.0
    W = rt.transport_weights(S)
    assert np.allclose(np.diag(W), 0.0) and np.allclose(np.abs(W).sum(1), 1.0)
    sc = rt.GramScorer(T, d)
    obs = sc.score_rows(W)
    ex = _explicit_scores(W @ T, T)
    for k in ("raw", "residual", "norm_pred"):
        assert np.allclose(obs[k], ex[k], atol=1e-9)
    # permutation null: same explicit identity with permuted weights
    Wp = rt.permute_offdiag(W, rng)
    assert np.allclose(np.sort(Wp, axis=1), np.sort(W, axis=1))
    exp = _explicit_scores(Wp @ T, T)
    got = sc.score_rows(Wp)
    assert np.allclose(got["residual"], exp["residual"], atol=1e-9)
    # covariance-matched null lives in the span and has the LOSO mean / covariance by construction
    rs = np.random.default_rng(3)
    c = rs.standard_normal((n, n))
    np.fill_diagonal(c, 0.0)
    s = np.sqrt(n - 2)
    G = rt.loso_mean(T)
    Xc = np.stack([G[w] + sum(c[w, j] * (T[j] - G[w]) for j in range(n) if j != w) / s for w in range(n)])
    got = sc.null_cov_matched(np.random.default_rng(3))
    assert np.allclose(got["residual"], _explicit_scores(Xc, T)["residual"], atol=1e-9)
    # rotated real target: norm preserved, lies in the span, and the coordinate formulas agree with explicit vectors
    Q, _ = np.linalg.qr(T.T)  # orthonormal basis of span(T) (n <= d, full rank)
    got = sc.null_rotated_target(np.random.default_rng(5))
    u = np.random.default_rng(5).standard_normal((n, sc.rank))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # explicit: rotate T_w within the span using the scorer's own basis B (T = Q_B B^T)
    lam, U = np.linalg.eigh(sc.G)
    keep = lam > lam.max() * 1e-10
    QB = T.T @ (U[:, keep] / np.sqrt(lam[keep])[None, :])  # d x r, orthonormal columns
    assert np.allclose(QB.T @ QB, np.eye(sc.rank), atol=1e-8)
    Xr = (u * np.linalg.norm(sc.B, axis=1)[:, None]) @ QB.T
    assert np.allclose(np.linalg.norm(Xr, axis=1), np.linalg.norm(T, axis=1))
    assert np.allclose(got["residual"], _explicit_scores(Xr, T)["residual"], atol=1e-8)
    # isotropic null: in a high dimension the raw cosine is centred at 0 with scale ~ 1/sqrt(d)
    big = rt.GramScorer(T, 100_000)
    vals = np.concatenate([big.null_isotropic(np.ones(n), np.random.default_rng(k))["raw"] for k in range(200)])
    assert abs(vals.mean()) < 0.002 and 0.002 < vals.std() < 0.005


def test_transport_recovers_planted_relations_and_permutation_null_does_not():
    rng = np.random.default_rng(2)
    n, k, d = 20, 3, 60
    C = rng.normal(size=(n, k))
    S0 = C @ rng.normal(size=(k, d)) + 0.1 * rng.normal(size=(n, d))  # template-free source: signed weights cancel
    T = 5.0 + C @ rng.normal(size=(k, d)) + 0.1 * rng.normal(size=(n, d))
    r = rt.transport(S0, T, rng, n_perm=50, n_draws=50)
    # the preregistered signed-kernel formula loses the template when sum_j s_j ~ 0 (documented); the centred variant does not
    assert abs(r["weights"]["signed_sum_mean"]) < 0.3 and r["observed"]["residual_cosine"]["mean"] < r["centred_variant"]["observed"]["residual_cosine"]["mean"]
    assert r["centred_variant"]["observed"]["residual_cosine"]["mean"] > 0.6
    for name, null in r["centred_variant"]["nulls"].items():
        assert null["residual_cosine"]["score_minus_null"]["mean"] > 0.4, name
        assert null["residual_cosine"]["score_minus_null"]["sign_flip_p"] < 0.01, name
    # with a template in the source (our fields), the signed-kernel formula recovers the relations too
    r2 = rt.transport(S0 + 5.0, T, rng, n_perm=50, n_draws=50)
    assert r2["weights"]["signed_sum_mean"] > 0.9 and r2["observed"]["residual_cosine"]["mean"] > 0.6
    for name, null in r2["nulls"].items():
        assert null["residual_cosine"]["score_minus_null"]["mean"] > 0.4, name
        assert null["residual_cosine"]["score_minus_null"]["sign_flip_p"] < 0.01, name
    assert r2["nulls"]["scene_permutation"]["residual_cosine"]["null_mean"] < 0.2
    # unrelated source, unstructured target: neither estimator beats its permutation null
    # (with a 3-d structured target the per-scene residual cosines are +-0.5, so that negative control would be noisy)
    T2 = 5.0 + rng.normal(size=(n, d))
    r0 = rt.transport(rng.normal(size=(n, d)) + 5.0, T2, rng, n_perm=50, n_draws=50)
    for blk in (r0, r0["centred_variant"]):
        sm = blk["nulls"]["scene_permutation"]["residual_cosine"]["score_minus_null"]
        assert abs(sm["mean"]) < 0.1 and sm["sign_flip_p"] > 0.01


def test_centred_weights_match_explicit_loso_centring():
    rng = np.random.default_rng(11)
    S = rng.normal(size=(7, 12)) + 1.0
    W = rt.centred_transport_weights(S)
    n = len(S)
    for w in range(n):
        m = np.mean([S[j] for j in range(n) if j != w], axis=0)
        cs = np.array([0.0 if j == w else rt.row_cos((S[w] - m)[None], (S[j] - m)[None])[0] for j in range(n)])
        assert np.allclose(W[w], cs / np.abs(cs).sum())
    A = rt.centred_rows(W)
    T = rng.normal(size=(7, 12))
    G = rt.loso_mean(T)
    explicit = np.stack([G[w] + sum(W[w, j] * (T[j] - G[w]) for j in range(n)) for w in range(n)])
    assert np.allclose(A @ T, explicit)


def test_residual_scoring_baselines():
    rng = np.random.default_rng(4)
    n, d = 10, 30
    T = 3.0 + rng.normal(size=(n, d))
    M = T + 0.3 * rng.normal(size=(n, d))
    donors = np.array([(i + 1) % n for i in range(n)])
    r = rt.residual_scoring(M, T, donors, [f"s{i}" for i in range(n)])
    assert np.allclose(r["per_scene"]["field_mean"]["residual"], 0.0)
    assert all(a != b for a, b in zip(r["scenes"], r["donor_scene"]))
    assert r["pooled"]["model"]["residual"]["mean"] > r["pooled"]["copy_delta"]["residual"]["mean"]
    assert r["pooled"]["copy_delta"]["raw"]["mean"] > 0.8  # the template makes another scene's field a strong raw match
    assert r["model_minus_baseline"]["copy_delta"]["residual"]["sign_flip_p"] < 0.05


def test_boot_mean_reproduces_cluster_bootstrap_mean_exactly():
    x = np.random.default_rng(7).normal(size=23)
    x[3] = np.nan
    a, b = rt.boot_mean(x), cluster_bootstrap_mean(x)
    assert a == b


def test_spearman_ties_and_block():
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 4.0])
    y = np.array([2.0, 1.0, 3.0, 3.0, 6.0, 5.0])
    assert np.allclose(rt.average_ranks(x), [1, 2.5, 2.5, 4, 6, 5])
    rx, ry = rt.average_ranks(x), rt.average_ranks(y)
    expect = np.corrcoef(rx, ry)[0, 1]
    assert abs(rt.spearman(x, y) - expect) < 1e-12
    blk = rt.spearman_block(x, y, np.random.default_rng(0), n_boot=200, n_perm=200)
    assert blk["ci_low"] <= blk["rho"] <= blk["ci_high"] and 0 < blk["p_perm"] <= 1
    assert np.allclose(rt.average_ranks_rows(np.stack([x, y]))[1], ry)


def test_characterise_field_single_direction_vs_spread():
    rng = np.random.default_rng(8)
    d = 50
    v = rng.normal(size=d)
    one = np.outer(1.0 + 0.1 * rng.normal(size=12), v)
    spread = rng.normal(size=(12, d))
    a = rt.characterise_field(one, rng, n_mc=20)
    b = rt.characterise_field(spread, rng, n_mc=20)
    assert a["spherical_variance"] < 0.01 and a["uniform_random_same_dim"]["z"] > 5
    assert b["pc1_variance_fraction"] < 0.4 and abs(b["uniform_random_same_dim"]["z"]) < 4
    assert 0.0 <= a["pc1_variance_fraction"] <= 1.0


def test_slim_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(9)
    n, T, D = 3, 2, 4
    full = {
        "cell_id": np.asarray(["a", "b", "c"]), "context": rng.normal(size=(n, 16, 16, D)).astype(np.float16),
        "target": rng.normal(size=(n, T, 16, 16, D)).astype(np.float16), "prediction": rng.normal(size=(n, T, 16, 16, D)).astype(np.float16),
        "zero_prediction": rng.normal(size=(n, T, 16, 16, D)).astype(np.float16), "actions": np.zeros((n, T, 7), np.float32),
        "token_mask": rng.integers(0, 2, size=(n, T + 1, 16, 16)).astype(bool), "has_mask": np.ones(n, bool),
    }
    src = tmp_path / "latent_cache.npz"
    np.savez_compressed(src, meta=json.dumps({"n_cells": n}), **full)
    dst = tmp_path / "latent_slim.npz"
    rt.slim_cache(src, dst)
    a, b = rt.load_latents(src), rt.load_latents(dst)
    for k in ("target_last", "prediction_last", "token_mask_last", "context_pooled"):
        assert np.allclose(a[k].astype(np.float32), b[k].astype(np.float32))
    assert a["n_steps"] == b["n_steps"] == T and list(b["cell_id"]) == ["a", "b", "c"]
    assert np.allclose(b["target_last"].astype(np.float32), full["target"][:, -1].astype(np.float32))


def test_synthetic_selftest_passes_and_writes_outputs(tmp_path):
    out = rt.self_test(tmp_path, n_perm=60, n_draws=60, seed=0)
    assert out["passed"], out["verdict"]
    st = json.loads((tmp_path / "relational_transport_selftest.json").read_text())
    assert st["interpretation_scope"] == rt.INTERPRETATION_SCOPE
    planted = json.loads((tmp_path / "relational_transport_synthetic_planted.json").read_text())
    assert planted["interpretation_scope"] == rt.INTERPRETATION_SCOPE
    g = planted["groups"]["hazard_corridor"]
    assert set(g["transport"]) == set(rt.TRANSPORT_BLOCKS) and g["n_scenes"] == 24
    # the physics field transports across arms only through the consequence-carrying identity (true A ped -> true B cone)
    swap = g["identity_swap"]["true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched)"]
    assert swap["swapped"]["observed"]["residual_cosine"]["mean"] > 0.5 > swap["matched"]["observed"]["residual_cosine"]["mean"]
    md = rt.markdown_summary(planted)
    assert "Relational transport" in md and "hazard_corridor" in md and "Identity swap" in md
    null = json.loads((tmp_path / "relational_transport_synthetic_null.json").read_text())
    ns = null["groups"]["hazard_corridor"]["identity_swap"]["true: A.ped_did -> B.cone_did (swapped) vs B.ped_did (matched)"]
    assert ns["swapped"]["nulls"]["scene_permutation"]["residual_cosine"]["score_minus_null"]["sign_flip_p"] > 0.05


def test_scene_tokens_from_cache_mask_and_covariates():
    arms = rt.make_synthetic(np.random.default_rng(0), n_scenes=6)
    a = arms["A"]
    cells = rt.scene_cells(a["rows"])
    toks, src = rt.scene_tokens(a["rows"], cells, None, a["latents"]["token_mask_last"], 3, "hazard_corridor")
    assert src == "cache_token_mask" and all(len(t) == 12 for t in toks.values())
    y = a["latents"]["target_last"]
    pid = sorted(cells)[0]
    v = rt.field_vector(y, cells[pid], toks[pid], rt.FIELD_DEFS["ped_did"])
    vt = rt.field_vector(y, cells[pid], toks[pid], rt.FIELD_DEFS["ped_did"], pool=False)
    assert v.shape == (y.shape[-1],) and np.allclose(v, vt.reshape(12, -1).mean(0))
    empty, src2 = rt.scene_tokens(a["rows"], cells, None, a["latents"]["token_mask_last"], 3, "hazard")
    assert empty == {} and src2 == "unavailable_without_masks"
    cov = rt.scene_covariates(a["rows"], cells, sorted(cells))
    assert set(cov) == set(rt.COVARIATES) and cov["hazard_patches"]["degenerate"] and not cov["hazard_pixels"]["degenerate"]
