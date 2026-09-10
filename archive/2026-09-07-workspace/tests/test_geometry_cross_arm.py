"""CPU tests for the cross-arm geometry map: Procrustes transport recovers a planted rotation; the Gram-trick capture
equals the SVD subspace capture; the 80 % quota separates a rank-2 code from noise; the relational residualisation
removes the AdaLN-style artifact and keeps the planted subspace; the residual random control lives in the action
complement; the planted synthetic dumps reproduce the four registered predictions and the planted-null dumps do not;
the JSON schema (ranked tables, registered_tests, interpretation_scope) is written."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import geometry_cross_arm as gx  # noqa: E402
from sonar_metrics import principal_angles_deg, svd_subspace  # noqa: E402


def test_orthogonal_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    d = 24
    W_true = np.linalg.qr(rng.normal(size=(d, d)))[0]
    X = rng.normal(size=(400, d))
    Y = X @ W_true + 0.05 * rng.normal(size=(400, d))
    W = gx.orthogonal_procrustes(X.T @ Y)
    assert np.allclose(W @ W.T, np.eye(d), atol=1e-10)
    assert np.linalg.norm(W - W_true) / np.sqrt(d) < 0.02
    # LOSO folds: held-out residual after transport is the noise floor, raw residual is not
    scene = np.repeat(np.arange(8), 50)
    folds, W_full, log = gx.transport_folds(X, Y, scene, list(range(8)))
    assert len(folds) == 8 and log["heldout_relative_residual_transported"] < 0.1 < log["heldout_relative_residual_raw"]
    # a direction transported with the fitted map lands where W_true puts it
    q = rng.normal(size=(d, 3))
    q = np.linalg.qr(q)[0]
    assert max(principal_angles_deg(W_full.T @ q, W_true.T @ q)) < 3.0


def test_capture_k_matches_svd_subspace_and_proj_cos():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 40))
    r = rng.normal(size=40)
    for k in (1, 2, 4, 8):
        Q = svd_subspace(X, k)
        assert abs(gx.capture_k(X, r, k) - gx.proj_cos(Q, r) ** 2) < 1e-9
    assert np.isnan(gx.capture_k(X, np.zeros(40), 2))


def test_quota_for_capture_separates_code_from_noise():
    rng = np.random.default_rng(2)
    d = 64
    S = np.linalg.qr(rng.normal(size=(d, 2)))[0].T
    code = (rng.normal(size=(16, 2)) * [3.0, 3.0]) @ S + 0.1 * rng.normal(size=(16, d))
    noise = 0.5 * rng.normal(size=(16, d))
    lam_code = np.linalg.svd(code, compute_uv=False) ** 2 / 16
    lam_noise = np.linalg.svd(noise, compute_uv=False) ** 2 / 16
    qc, qn = gx.quota_for_capture(lam_code, d, 0.8), gx.quota_for_capture(lam_noise, d, 0.8)
    assert qc["hard_rank"] <= 2 and qn["hard_rank"] >= 8
    assert qc["quota"] < qn["quota"]
    assert abs(qc["capture_at_alpha"] - 0.8) < 1e-3 and abs(qn["capture_at_alpha"] - 0.8) < 1e-3


def _rows(rng, d, S, S_rel, a_dir, gamma_ped, gamma_cone, gamma_null, rel_on):
    """Contrast rows with an AdaLN-style hazard-scaled action artifact in every level and a rank-2 relational term on
    the requested identity; the H0' DiD carries the artifact only."""
    rows = {k: np.zeros((S, d)) for k in ("app_ped", "app_cone", "app_ped_a0", "app_cone_a0", "did_ped", "did_cone", "did_null", "action")}
    for i in range(S):
        g = 2.0 + 0.2 * rng.normal()
        rows["action"][i] = g * a_dir + 0.05 * rng.normal(size=d)
        rel = (rng.normal(size=2) * [3.0, 2.0]) @ S_rel
        rows["did_ped"][i] = gamma_ped * g * a_dir + (rel if rel_on == "ped" else 0) + 0.05 * rng.normal(size=d)
        rows["did_cone"][i] = gamma_cone * g * a_dir + (rel if rel_on == "cone" else 0) + 0.05 * rng.normal(size=d)
        rows["did_null"][i] = gamma_null * g * a_dir + 0.05 * rng.normal(size=d)
    return rows


def test_relational_rows_remove_artifact_and_keep_planted_subspace():
    rng = np.random.default_rng(3)
    d, S = 48, 14
    B = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    S_rel, a_dir = B[:2], B[2]
    rows = _rows(rng, d, S, S_rel, a_dir, 0.5, 0.5, 0.5, "ped")
    rel, Q_act = gx.relational_rows(rows, np.arange(S), k_action=2)
    # the a_dir component is gone from both, the planted subspace survives in rel_ped only
    assert np.abs(rel["rel_ped"] @ a_dir).max() < 0.05 and np.abs(rel["rel_cone"] @ a_dir).max() < 0.05
    ped_energy = np.linalg.norm(rel["rel_ped"] @ S_rel.T, axis=1) ** 2 / np.linalg.norm(rel["rel_ped"], axis=1) ** 2
    cone_energy = np.linalg.norm(rel["rel_cone"] @ S_rel.T, axis=1) ** 2 / np.linalg.norm(rel["rel_cone"], axis=1) ** 2
    assert ped_energy.mean() > 0.9 and cone_energy.mean() < 0.2
    # an artifact that differs between the in-lane level and H0' survives the null subtraction but not the projection
    rows2 = _rows(rng, d, S, S_rel, a_dir, 0.8, 0.8, 0.3, "ped")
    rel2, _ = gx.relational_rows(rows2, np.arange(S), k_action=2)
    assert np.abs(rel2["rel_cone"] @ a_dir).max() < 0.05
    # the residual random control is orthogonal to the action subspace and orthonormal
    Qr = gx.random_subspace(d, 4, np.random.RandomState(0))
    Qres = gx.residual_random_subspace(Qr, Q_act)
    assert np.allclose(Qres.T @ Qres, np.eye(4), atol=1e-10) and np.abs(Qres.T @ Q_act).max() < 1e-10


def test_dd_refit_statistic_matches_scores_and_flips_sign_under_relabelling():
    rng = np.random.default_rng(4)
    d, S, k = 32, 10, 2
    fold_rows = {arm: {t: rng.normal(size=(S, S, d)) for t in gx.RELATIONAL_TYPES} for arm in ("A", "B")}
    rand_cap = {arm: {t: rng.uniform(0, 0.1, size=S) for t in gx.RELATIONAL_TYPES} for arm in ("A", "B")}
    obs = gx.dd_refit_statistic(fold_rows, rand_cap, k, None)
    # identity relabelling reproduces the per-scene definition
    for i in range(S):
        tr = np.asarray([j for j in range(S) if j != i])
        val = {}
        for arm in ("A", "B"):
            Rp, Rc = fold_rows[arm]["rel_ped"][i], fold_rows[arm]["rel_cone"][i]
            val[arm] = (gx.capture_k(Rp[tr], Rp[i], k) - rand_cap[arm]["rel_ped"][i]) - (gx.capture_k(Rc[tr], Rc[i], k) - rand_cap[arm]["rel_cone"][i])
        assert abs(obs[i] - (val["A"] - val["B"])) < 1e-9
    # relabelling every scene flips the sign of every DD_i exactly (both arms swapped)
    flipped = gx.dd_refit_statistic(fold_rows, rand_cap, k, -np.ones(S))
    assert np.allclose(flipped, -obs)
    null = gx.dd_refit_null(fold_rows, rand_cap, k, gx.sign_flip_matrix(S, 64, np.random.default_rng(0)))
    assert null["null_t"].shape == (64,) and 0 < null["p_raw_one_sided"] <= 1


@pytest.fixture(scope="module")
def planted_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("planted")
    rep = gx.main(["--synthetic", "--out", str(out), "--synthetic-scenes", "14", "--synthetic-dim", "48", "--synthetic-sites", "L03.attn_out", "L07.attn_out",
                   "--steps", "0", "--groups", "hazard", "corridor", "--n-boot", "300", "--n-perm", "199", "--n-perm-refit", "99"])
    return rep, out


def test_planted_synthetic_reproduces_the_four_predictions(planted_report):
    rep, out = planted_report
    assert rep["self_test"]["passed"] and all(rep["self_test"]["checks"].values())
    reg = rep["registered_tests"]
    site = reg["i_appearance_shared"]["s0|hazard"]["L03.attn_out"]
    # (i) brake-cell appearance: transported ~ within (>> random) in both directions, raw ambient overlap at chance
    for t in ("app_ped_a0", "app_cone_a0"):
        for d in ("A_to_B", "B_to_A"):
            x = site[t][d]
            assert x["transported"]["mean"] > 0.85 and x["transported_minus_random"]["ci_low"] > 0.3
            assert abs(x["transported_minus_within"]["mean"]) < 0.1
        assert site[t]["A_to_B"]["raw_minus_random"]["ci_high"] < 0.2
        assert max(site[t]["principal_angles_deg"]["full_fit_transported"][:2]) < 15 < min(site[t]["principal_angles_deg"]["full_fit_raw"][:2])
    # the action-averaged main effect of the solid identity is contaminated by DiD/2 and is reported, not registered
    assert site["app_ped"]["registered"] is False and site["app_ped_a0"]["registered"] is True
    # (ii) double dissociation with the refit permutation null, max-T over the two sites
    ii = reg["ii_identity_attachment"]["s0|hazard"]["sites"]["L03.attn_out"]
    assert ii["ped_minus_cone_A"]["ci_low"] > 0.3 and ii["ped_minus_cone_B"]["ci_high"] < -0.3
    assert ii["double_dissociation"]["p_maxt_fwer"] < 0.05 and ii["double_dissociation"]["refit_matches_scores"]
    # (iii) A's pedestrian-relational subspace transported into B: chance on B's pedestrian cells, above chance on cone cells
    iii = reg["iii_transport_identity_remap"]["s0|hazard"]["sites"]["L03.attn_out"]
    assert iii["A_ped_to_B_cone_minus_random"]["ci_low"] > 0.3 and abs(iii["A_ped_to_B_ped_minus_random"]["mean"]) < 0.1
    assert iii["B_cone_to_A_ped_minus_random"]["ci_low"] > 0.3 and abs(iii["B_cone_to_A_cone_minus_random"]["mean"]) < 0.1
    assert iii["conceptor_cross_similarity"]["B_rel_cone"] > iii["conceptor_cross_similarity"]["B_rel_ped"]
    # (iv) quota for 80 % capture: compact for the solid identity, diffuse for the ghost
    iv = reg["iv_rank_quota"]["s0|hazard"]["sites"]["L03.attn_out"]
    assert iv["A"]["rel_ped"]["quota"] < iv["A"]["rel_cone"]["quota"] and iv["B"]["rel_cone"]["quota"] < iv["B"]["rel_ped"]["quota"]
    assert iv["A"]["rel_ped"]["hard_rank"] <= 3 <= iv["A"]["rel_cone"]["hard_rank"]


def test_output_schema_and_ranked_tables(planted_report):
    rep, out = planted_report
    path = out / "synthetic" / "out" / "cross_arm_map.json"
    assert path.exists()
    j = json.loads(path.read_text())
    assert j["protocol"] == gx.PROTOCOL and "causal claims come from patching" in j["interpretation_scope"]
    assert set(j["registered_tests"]) >= {"i_appearance_shared", "ii_identity_attachment", "iii_transport_identity_remap", "iv_rank_quota"}
    assert set(j["ranked_tables"]) == {"s0|hazard", "s0|corridor"}
    rows = j["ranked_tables"]["s0|hazard"]
    assert [r["site_id"] for r in rows] == sorted((r["site_id"] for r in rows), key=lambda s: -next(r["dd_t"] for r in rows if r["site_id"] == s))
    assert {"dd_t", "dd_p_maxt", "dd_p_refit_raw", "app_ped_a0_trans_AB_minus_within_B", "rel_ped_grassmann_transported", "quota80"} <= set(rows[0])
    e = next(e for e in j["entries"] if e["status"] == "ok")
    assert set(e["analysis"]["subspaces"]) == set(gx.TYPES) and "k4" in e["analysis"]["subspaces"]["rel_ped"]
    assert "_per_scene" not in e["analysis"]["subspaces"]["rel_ped"]["k4"] and "null_t" not in e["analysis"]["dd_refit"]
    assert j["transport"]["per_site_step"]["L03.attn_out|s0"]["heldout_relative_residual_transported"] < j["transport"]["per_site_step"]["L03.attn_out|s0"]["heldout_relative_residual_raw"]
    assert j["alignment"]["null_factor"] == "full" and j["n_scenes"] == 14


def test_planted_null_does_not_show_the_predictions(tmp_path):
    rep = gx.main(["--synthetic", "--synthetic-null", "--out", str(tmp_path), "--synthetic-scenes", "14", "--synthetic-dim", "48", "--synthetic-sites", "L03.attn_out",
                   "--steps", "0", "--groups", "hazard", "--n-boot", "300", "--n-perm", "199", "--n-perm-refit", "99"])
    assert rep["self_test"]["planted"] is False and rep["self_test"]["passed"], rep["self_test"]
    j = json.loads((tmp_path / "synthetic_null" / "out" / "cross_arm_map.json").read_text())
    reg = j["registered_tests"]
    site = reg["i_appearance_shared"]["s0|hazard"]["L03.attn_out"]
    # independent appearance subspaces: within-arm high, transported at chance
    assert site["app_ped_a0"]["A_to_B"]["within"]["mean"] > 0.85 and site["app_ped_a0"]["A_to_B"]["transported_minus_within"]["ci_high"] < -0.3
    assert site["app_ped_a0"]["pattern_consistent"] is False
    iii = reg["iii_transport_identity_remap"]["s0|hazard"]["sites"]["L03.attn_out"]
    assert iii["pattern_consistent"] is False and abs(iii["A_ped_to_B_cone_minus_random"]["mean"]) < 0.15
    ii = reg["ii_identity_attachment"]["s0|hazard"]["sites"]["L03.attn_out"]
    assert abs(ii["ped_minus_cone_A"]["mean"]) < 0.15 and abs(ii["double_dissociation"]["mean"]) < 0.2
    assert j["self_test"]["checks"] == {"i": True, "ii": True, "iii": True, "iv": True}
