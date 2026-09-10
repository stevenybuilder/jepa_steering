"""CPU tests for the soft-conceptor entrant: planted rank-3 interaction subspace inside high-dim noise with a
dominant action direction; the conceptor recovers it, AND-NOT removes the action direction, rank-one does not,
matched-spectrum random has ~0 similarity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))
sys.path.insert(0, str(ROOT / "tests"))

import geometry_conceptor as gc  # noqa: E402
from geometry_models import (  # noqa: E402
    ConceptorSubspace,
    alpha_for_quota,
    conceptor_and_not,
    conceptor_apply,
    conceptor_factored,
    conceptor_matrix,
    conceptor_quota,
    conceptor_similarity,
    matched_spectrum_random,
    rank_one_projector,
)
from test_geometry_localize import D, build_dump, build_stimulus  # noqa: E402


def planted(n_scenes=40, seed=0, action_gain=6.0, noise=0.15):
    """delta_i = S c_i (rank 3) + g_i a (dominant shared action direction) + noise; a_i = a + small noise."""
    rng = np.random.default_rng(seed)
    S = np.linalg.qr(rng.normal(size=(D, 4)))[0].T
    a = S[3]
    S = S[:3]
    C = rng.normal(size=(n_scenes, 3)) * np.array([3.0, 2.0, 1.5])
    delta = C @ S + np.outer(action_gain * (1 + 0.2 * rng.normal(size=n_scenes)), a) + rng.normal(size=(n_scenes, D)) * noise
    act = np.outer(4.0 * (1 + 0.1 * rng.normal(size=n_scenes)), a) + rng.normal(size=(n_scenes, D)) * noise
    return delta, act, S, a


def proj(S):
    return {"eigvecs": S, "mu": np.ones(len(S)), "lam": np.ones(len(S))}


def test_conceptor_closed_form_matches_eigendecomposition():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 12))
    c = conceptor_factored(X, 3.0)
    Xc = X - X.mean(0)
    R = Xc.T @ Xc / len(X)
    C_ref = R @ np.linalg.inv(R + (3.0 ** -2) * np.eye(12))
    assert np.allclose(conceptor_matrix(c), C_ref, atol=1e-8)
    assert abs(conceptor_quota(c, 12) - np.trace(C_ref) / 12) < 1e-8
    # similarity is the matrix cosine
    c2 = conceptor_factored(X[:15], 10.0)
    C2 = conceptor_matrix(c2)
    ref = np.trace(C_ref @ C2) / np.sqrt(np.trace(C_ref @ C_ref) * np.trace(C2 @ C2))
    assert abs(conceptor_similarity(c, c2) - ref) < 1e-8
    # AND-NOT vs the closed form on a full-rank case (inverses well defined), eigenvalues in [0, 1]
    A = conceptor_factored(rng.normal(size=(40, 12)), 5.0)
    B = conceptor_factored(rng.normal(size=(40, 12)), 5.0)
    andnot_ref = np.linalg.inv(np.linalg.inv(conceptor_matrix(A)) + np.linalg.inv(np.eye(12) - conceptor_matrix(B)) - np.eye(12))
    got = conceptor_matrix(conceptor_and_not(A, B))
    assert np.allclose(got, andnot_ref, atol=1e-5)
    ev = np.linalg.eigvalsh(got)
    assert ev.min() > -1e-9 and ev.max() < 1 + 1e-9
    # target-quota aperture
    al = alpha_for_quota(c["lam"], 12, 0.3)
    assert abs(conceptor_quota(conceptor_factored(X, al), 12) - 0.3) < 1e-3


def test_planted_subspace_recovered_and_action_removed():
    delta, act, S, a = planted()
    scale = np.sqrt(np.mean(np.sum(delta**2, 1)))
    Ds, As = delta / scale, act / scale
    alpha = alpha_for_quota(conceptor_factored(Ds, 1.0, center=False)["lam"], D, 4.0 / D)
    cs = gc.fit_conceptor_set(Ds, As, alpha, seed=0)
    P = proj(S)
    # C_int contains S but also the action direction; AND-NOT C_action removes it
    assert conceptor_similarity(cs["safety"], P) > 0.8, conceptor_similarity(cs["safety"], P)
    assert np.linalg.norm(conceptor_apply(a[None], cs["safety"])) < 0.15
    assert np.linalg.norm(conceptor_apply(a[None], cs["int"])) > 0.8
    # rank-one mean-difference projector is dominated by the action direction
    assert conceptor_similarity(cs["rank_one"], P) < 0.3
    assert abs(float(cs["rank_one"]["eigvecs"][0] @ a)) > 0.9
    # matched-spectrum random control: same quota, ~0 similarity to the planted subspace
    assert abs(conceptor_quota(cs["random"], D) - conceptor_quota(cs["safety"], D)) < 1e-9
    assert conceptor_similarity(cs["random"], P) < 0.15
    assert conceptor_similarity(cs["random"], cs["safety"]) < 0.15
    # stability of refits
    st = gc.stability_refits(Ds, As, alpha)
    assert st["safety_mean"] > 0.85


def test_conceptor_entrant_reads_out_interaction_and_ignores_action():
    """Cells built from planted scene effects; the entrant's readout must recover the interaction label."""
    rng = np.random.default_rng(2)
    delta, act, S, a = planted(n_scenes=40)
    n = 40 * 4
    pair, hz, ac, X = [], [], [], np.zeros((n, D))
    i = 0
    for s in range(40):
        H = rng.normal(size=D) * 0.5
        for h in (0, 1):
            for aa in (0, 1):
                pair.append(f"p{s}"); hz.append(h); ac.append(aa)
                X[i] = h * H + aa * act[s] + (2 * h - 1) * (2 * aa - 1) / 4 * delta[s] + rng.normal(size=D) * 0.05
                i += 1
    meta = {"pair_id": np.asarray(pair), "hazard": np.asarray(hz), "action": np.asarray(ac)}
    y = ((np.asarray(hz) * 2 - 1) * (np.asarray(ac) * 2 - 1) > 0).astype(float)
    from geometry_models import quartet_residualize, auroc
    R = quartet_residualize(X, meta, keep_hazard=True)
    m = ConceptorSubspace(budget=4, lam=1.0, target_dims=4.0).fit(R, y, meta)
    g = m.geometry_summary()
    # the tournament hands this entrant quartet-residualized features for the interaction target, so the
    # action vectors are ~0 there and AND-NOT leaves C_int essentially unchanged
    assert g["kind"] == "conceptor" and 0 < g["quota_safety"] <= g["quota_int"] + 1e-6
    assert auroc(m.score(R), y) > 0.9
    # with the action effect already residualized out of the rows, C_safety ~ C_int, which still holds the
    # planted subspace (plus the action direction that the deltas themselves carry); the AND-NOT removal of
    # the action direction is exercised in test_planted_subspace_recovered_and_action_removed
    assert conceptor_similarity(m.c_safety, proj(S)) > 0.7


def test_end_to_end_on_fake_dump(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, 12)
    rng = np.random.default_rng(3)
    delta, act, S, a = planted(n_scenes=12, seed=3)
    X = np.zeros((len(cells), D))
    Hs = {s: rng.normal(size=D) for s in range(12)}
    for c in cells:
        s = int(c["seed"]) - 300
        h, aa = c["hazard"], c["candidate_action"]
        X[c["row"]] = 2.0 * Hs[s] * h + aa * act[s] + (2 * h - 1) * (2 * aa - 1) / 4 * delta[s] + rng.normal(size=D) * 0.05
    dump = build_dump(tmp_path, stim, cells, X, sites=("L07.attn_out", "L08.attn_out"))
    seeds = tmp_path / "disc.txt"
    seeds.write_text("\n".join(str(300 + s) for s in range(12)))
    rep = gc.main(["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "out"), "--discovery-seeds", str(seeds),
                   "--steps", "0", "1", "--groups", "egg", "--n-boot", "50", "--sweep-readout-alphas"])
    e = [x for x in rep["entries"] if x["site_id"] == "L07.attn_out" and x["step"] == 0][0]
    assert e["status"] == "ok" and e["n_scenes"] == 12
    assert set(e["sweep"]) == {"1.0", "3.0", "10.0", "30.0", "100.0"}
    assert e["quotas"]["safety"] <= e["quotas"]["int"] + 1e-9
    r = e["readout"]["interaction"]
    assert r["status"] == "ok" and r["safety"]["point"] > 0.8
    assert r["safety_minus_random"]["point"] > 0.0
    # export format
    with np.load(e["conceptor_file"]) as z:
        keys = set(z.files)
        for k in ("safety_eigvecs", "safety_mu", "int_eigvecs", "int_mu", "action_eigvecs", "action_mu", "random_eigvecs", "random_mu", "rank_one_direction", "mean", "scale", "alpha", "quota_safety", "token_pool"):
            assert k in keys, k
        V, mu = z["safety_eigvecs"].astype(np.float64), z["safety_mu"].astype(np.float64)
        Cs = V.T @ np.diag(mu) @ V
        assert np.allclose(Cs, Cs.T) and np.all(np.linalg.eigvalsh(Cs) > -1e-6)
        assert V.shape[1] == D
    cf = Path(e["conceptor_file"]); side = json.loads((cf.parent / (cf.name[:-4] + ".json")).read_text())
    assert cf.name == "L07.attn_out__s0__egg.npz"
    assert "sweep" in side and "apply" in side
    # cross-step / adjacent-layer similarity tables exist
    assert "L07.attn_out|egg|s0->s1" in rep["cross_similarity"]["across_steps"]
    assert "L07.attn_out->L08.attn_out|s0|egg" in rep["cross_similarity"]["across_adjacent_layers"]
    assert "s0|egg" in rep["quota_tables"] and "L07.attn_out" in rep["quota_tables"]["s0|egg"]
