"""CPU tests for the geometry tournament: planted geometries must be recovered by the right model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import geometry_frames as gf  # noqa: E402
import geometry_tournament as gt  # noqa: E402
from geometry_models import (  # noqa: E402
    CircularFactorized,
    CounterfactualPairDirection,
    JacobianLocal,
    LinearMeanDiff,
    LinearRidge,
    LowRankSubspace,
    ManifoldKNN,
    TranscoderPlaceholder,
    auroc,
    cluster_bootstrap,
    effective_rank,
    mahalanobis_diag,
    on_manifold_scores,
    r_squared,
    random_equal_norm_edits,
    scene_center,
    scene_folds,
    subspace_similarity,
    write_edit_artifact,
)

D = 32


def factorial_meta(n_scenes: int, seed_offset: int = 200) -> dict[str, np.ndarray]:
    pair, hz, ac, seed = [], [], [], []
    for s in range(n_scenes):
        for h in (0, 1):
            for a in (0, 1):
                pair.append(f"contact_seed{seed_offset + s:06d}")
                hz.append(h)
                ac.append(a)
                seed.append(seed_offset + s)
    return {
        "pair_id": np.asarray(pair),
        "hazard": np.asarray(hz),
        "action": np.asarray(ac),
        "seed": np.asarray(seed),
    }


def planted_linear(n_scenes=40, rng=None, main_effects=False):
    rng = rng or np.random.default_rng(0)
    meta = factorial_meta(n_scenes)
    n = len(meta["pair_id"])
    w = rng.normal(size=D)
    w /= np.linalg.norm(w)
    scene_offset = rng.normal(size=(n_scenes, D)) * 2.0  # nuisance per scene
    X = np.repeat(scene_offset, 4, axis=0) + rng.normal(size=(n, D)) * 0.3
    interaction = (meta["hazard"] * meta["action"]).astype(float)
    y = 3.0 * interaction + rng.normal(size=n) * 0.1
    X = X + np.outer(y, w)
    if main_effects:  # hazard and action main effects on their own axes (as real activations would carry)
        X = X + np.outer(meta["hazard"], rng.normal(size=D)) + np.outer(meta["action"], rng.normal(size=D))
    meta["angle"] = np.where(meta["hazard"] == 1, 0.0, np.pi)
    return X, y, meta, w


def planted_circular(n_scenes=60, rng=None):
    """Activations live on a circle parameterised by a bearing angle; label is a nonlinear function of angle."""
    rng = rng or np.random.default_rng(1)
    meta = factorial_meta(n_scenes)
    n = len(meta["pair_id"])
    theta = rng.uniform(-np.pi, np.pi, size=n)
    basis = np.linalg.qr(rng.normal(size=(D, 2)))[0].T  # 2 x D orthonormal
    X = 5.0 * (np.cos(theta)[:, None] * basis[0] + np.sin(theta)[:, None] * basis[1]) + rng.normal(size=(n, D)) * 0.2
    # contact if the bearing is within +-45 degrees of the path OR its antipode (non-linear in cos/sin)
    y = (np.abs(np.cos(2 * theta)) > np.cos(np.pi / 4)).astype(float) * (np.cos(2 * theta) > 0)
    meta["angle"] = theta
    return X, y, meta, basis


def planted_lowrank(n_scenes=40, rank=3, rng=None):
    rng = rng or np.random.default_rng(2)
    meta = factorial_meta(n_scenes)
    n = len(meta["pair_id"])
    basis = np.linalg.qr(rng.normal(size=(D, rank)))[0].T
    coords = rng.normal(size=(n, rank))
    y = coords @ np.array([1.0, -0.5, 0.25])[:rank] + rng.normal(size=n) * 0.05
    X = coords @ basis + rng.normal(size=(n, D)) * 0.05
    meta["angle"] = np.where(meta["hazard"] == 1, 0.0, np.pi)
    return X, y, meta, basis


def planted_manifold(n_scenes=60, rng=None):
    """Label depends on position along a curved 1-d manifold (XOR-like in any single axis)."""
    rng = rng or np.random.default_rng(3)
    meta = factorial_meta(n_scenes)
    n = len(meta["pair_id"])
    t = rng.uniform(0, 4 * np.pi, size=n)
    basis = np.linalg.qr(rng.normal(size=(D, 3)))[0].T
    X = (
        np.cos(t)[:, None] * basis[0] + np.sin(t)[:, None] * basis[1] + (t / (4 * np.pi))[:, None] * basis[2] * 2
    ) * 3.0 + rng.normal(size=(n, D)) * 0.1
    y = (np.sin(2 * t) > 0).astype(float)  # alternates 4 times along the curve
    meta["angle"] = t
    return X, y, meta


def cv_metric(model_fn, X, y, meta, kind, n_folds=5):
    scores = gt.cross_fit_scores(model_fn, X, y, meta, n_folds, seed=0)
    ok = np.isfinite(scores)
    return auroc(scores[ok], y[ok]) if kind == "binary" else r_squared(scores[ok], y[ok])


# ----------------------------------------------------------------------------- metrics


def test_auroc_and_r2_basics():
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert np.isnan(auroc(np.array([1.0, 2.0]), np.array([1, 1])))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == 1.0
    assert abs(r_squared(np.full(4, y.mean()), y)) < 1e-12


def test_scene_folds_keep_scenes_intact():
    meta = factorial_meta(10)
    folds = scene_folds(meta["pair_id"], 5, seed=0)
    assert sum(len(f) for f in folds) == 40
    for f in folds:
        assert len(f) % 4 == 0
        assert len(set(meta["pair_id"][f])) == len(f) // 4


def test_cluster_bootstrap_ci_contains_point():
    meta = factorial_meta(20)
    vals = np.random.default_rng(0).normal(size=80)
    res = cluster_bootstrap(lambda idx: float(vals[idx].mean()), meta["pair_id"], n_boot=200, seed=0)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]


# ----------------------------------------------------------------------------- planted geometries


def test_linear_direction_recovered_by_linear_models_and_stable():
    X, y, meta, w = planted_linear()
    # mean-difference recovers the *encoding* direction; a ridge readout is the
    # covariance-whitened direction, so only its predictions are checked.
    m = LinearMeanDiff().fit(X, y, meta)
    assert abs(m.direction @ w) > 0.9
    # raw features carry a large per-scene offset: a ridge readout does not transfer to new scenes
    assert cv_metric(lambda: LinearRidge(lam=1.0), X, y, meta, "continuous") < 0.5
    # within-scene coordinates remove the offset without using labels
    Xc = scene_center(X, meta["pair_id"])
    r2 = cv_metric(lambda: LinearRidge(lam=1.0), Xc, y, meta, "continuous")
    assert r2 > 0.9
    stab = gt.stability(lambda: LinearRidge(lam=1.0), Xc, y, meta, [0, 1, 2])
    assert stab["mean_similarity"] > 0.8  # ridge readouts on 80% scene resamples; gate is 0.70


def test_circular_geometry_recovered_only_by_circular_model():
    X, y, meta, basis = planted_circular()
    circ = CircularFactorized(budget=2, lam=1e-3).fit(X, y, meta)
    assert circ.circularity_r2 > 0.9
    assert subspace_similarity(circ.A, basis) > 0.95
    # the same fit on linear-planted data must NOT look circular
    Xl, yl, ml, _ = planted_linear()
    circ_l = CircularFactorized(budget=2, lam=1e-3).fit(Xl, yl, ml)
    assert circ_l.circularity_r2 < 0.3
    assert circ.phase_r2 > 0.95  # decoded phase matches the supervising bearing
    # label is a second-harmonic function of angle: the circular chart with 4 harmonic
    # coordinates reads it out; a linear direction on the (cos, sin) plane cannot
    au_circ = cv_metric(lambda: CircularFactorized(budget=4, lam=1e-3), X, y, meta, "binary")
    au_lin = cv_metric(lambda: LinearRidge(lam=1e-3), X, y, meta, "binary")
    au_lr = cv_metric(lambda: LowRankSubspace(budget=4, lam=1e-3), X, y, meta, "binary")
    assert au_circ > 0.9
    assert au_lin < 0.7 and au_lr < 0.7


def test_lowrank_subspace_recovered_with_correct_rank():
    X, y, meta, basis = planted_lowrank(rank=3)
    m = LowRankSubspace(budget=3, lam=1e-2).fit(X, y, meta)
    assert subspace_similarity(m.basis, basis) > 0.9
    assert effective_rank(X, 0.95) == 3
    r2 = cv_metric(lambda: LowRankSubspace(budget=3, lam=1e-2), X, y, meta, "continuous")
    assert r2 > 0.9
    # the chart's reported rank is the planted rank; a 1-d entrant reports rank 1
    assert m.geometry_summary()["rank"] == 3
    assert LinearMeanDiff().fit(X, y, meta).geometry_summary()["rank"] == 1
    # a rank-1 subspace explains far less of the label-weighted variance than rank 3
    m1 = LowRankSubspace(budget=1, lam=1e-2).fit(X, y, meta)
    assert sum(m1.explained) < sum(m.explained) - 0.2


def test_jacobian_local_uses_within_scene_differences():
    X, y, meta, w = planted_linear()
    diffs = JacobianLocal.within_scene_differences(X, meta)
    assert diffs.shape == (40 * 5, D)  # 2 hazard diffs + 2 action diffs + 1 interaction per scene
    m = JacobianLocal(budget=2, lam=1.0).fit(X, y, meta)
    # the tangent basis is built from within-scene differences, so it finds the planted
    # direction even though the between-scene offset (2.0 std) dominates raw variance
    assert abs(m.basis[0] @ w) > 0.9
    Xc = scene_center(X, meta["pair_id"])
    r2 = cv_metric(lambda: JacobianLocal(budget=2, lam=1.0), Xc, y, meta, "continuous")
    assert r2 > 0.9


def test_manifold_knn_beats_linear_on_curved_label():
    X, y, meta = planted_manifold()
    au_knn = cv_metric(lambda: ManifoldKNN(budget=8), X, y, meta, "binary")
    au_lin = cv_metric(lambda: LinearRidge(lam=1.0), X, y, meta, "binary")
    assert au_knn > 0.9
    assert au_lin < 0.7


def test_edit_directions_move_readout_toward_target():
    X, y, meta, _ = planted_linear()
    models = [
        LinearMeanDiff().fit(X, y, meta),
        LinearRidge(lam=1.0).fit(X, y, meta),
        LowRankSubspace(budget=3, lam=1.0).fit(X, y, meta),
        JacobianLocal(budget=2, lam=1.0).fit(X, y, meta),
        ManifoldKNN(budget=6).fit(X, y, meta),
    ]
    X = scene_center(X, meta["pair_id"])
    models = [m.fit(X, y, meta) for m in models]
    low = np.flatnonzero(y < 0.5)[0]  # a safe cell
    for m in models:
        delta = m.edit_direction(X[low], target=3.0, norm=1.0)
        assert delta.shape == (D,)
        assert abs(np.linalg.norm(delta) - 1.0) < 1e-6
        before = float(m.score(X[low][None])[0])
        after = float(m.score((X[low] + 3.0 * delta)[None])[0])
        assert after > before, m.name


def test_transcoder_placeholder_refuses():
    with pytest.raises(NotImplementedError):
        TranscoderPlaceholder().fit(np.zeros((4, D)), np.zeros(4), {})


# ----------------------------------------------------------------------------- frames


def look_at_matrix(cam_pos, target):
    """MuJoCo camera convention: -z looks at target, +y up."""
    z = cam_pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def fake_geometry(seed: int, hazard: int, action: int) -> dict:
    """Scene layout: eef hovers 7.5 cm above the H1 egg spot; H0 egg is 15 cm off-path at a per-scene bearing."""
    rng = np.random.default_rng(int(seed))
    h1 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.3, 0.6), 0.92])
    bearing = rng.uniform(-np.pi, np.pi)
    egg = h1 if hazard else h1 + 0.15 * np.array([np.cos(bearing), np.sin(bearing), 0.0])
    eef = h1 + np.array([0.0, 0.0, gf.HOVER_OFFSET_M])
    euler = np.array([np.pi, 0.0, rng.uniform(-np.pi, np.pi)])  # gripper pointing down, random yaw
    proprios = np.stack([np.concatenate([eef + 0.01, euler, [0.0]]), np.concatenate([eef, euler, [0.0]])])
    actions = np.zeros((3, 7))
    actions[:, 2] = -0.02 * (1 + action)  # descend, faster for the aggressive candidate
    cam_pos = np.array([0.4, 0.55, 1.4]) + rng.normal(size=3) * 0.02
    cam_mat = look_at_matrix(cam_pos, h1)
    egg_px_c = gf.project_to_pixels(egg, cam_pos, cam_mat, 85.0, (256, 256))
    egg_px = egg_px_c + np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    eef_px = gf.project_to_pixels(eef, cam_pos, cam_mat, 85.0, (256, 256))
    return {"egg": egg, "eef": eef, "euler": euler, "proprios": proprios, "actions": actions,
            "cam_pos": cam_pos, "cam_mat": cam_mat, "egg_px": egg_px, "eef_px": eef_px, "bearing": bearing}


def test_projection_and_euler_conventions():
    cam_pos = np.array([0.0, -1.0, 2.0])
    target = np.array([0.0, 0.0, 0.0])
    M = look_at_matrix(cam_pos, target)
    uv = gf.project_to_pixels(target, cam_pos, M, 90.0, (256, 256))
    assert np.allclose(uv, [128, 128])
    # a point to the camera's right and up maps right and up (v decreases)
    uv2 = gf.project_to_pixels(target + 0.5 * M[:, 0] + 0.5 * M[:, 1], cam_pos, M, 90.0, (256, 256))
    assert uv2[0] > 128 and uv2[1] < 128
    R = gf.euler_xyz_to_matrix([0.0, 0.0, np.pi / 2])
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_path_frame_separates_hazard_cells():
    g1 = fake_geometry(7, 1, 1)
    g0 = fake_geometry(7, 0, 1)
    c1 = gf.frame_coordinates(gf.CellGeometry("a", g1["egg"], g1["eef"], g1["euler"], g1["actions"], (256, 256),
                                                g1["cam_pos"], g1["cam_mat"], 85.0, g1["eef_px"], g1["egg_px"]))
    c0 = gf.frame_coordinates(gf.CellGeometry("b", g0["egg"], g0["eef"], g0["euler"], g0["actions"], (256, 256),
                                                g0["cam_pos"], g0["cam_mat"], 85.0, g0["eef_px"], g0["egg_px"]))
    assert c1[0]["hazard_under_path"][0] < 1e-6  # on the descent line
    assert abs(c0[0]["hazard_under_path"][0] - 0.15) < 1e-6
    assert abs(np.angle(np.exp(1j * (c0[1]["bearing"] - g0["bearing"])))) < 1e-9
    assert c1[2]["reprojection_error_px"] < 1e-6  # egg_px was generated with the same camera
    assert set(c1[0]) == set(gf.FRAME_NAMES)


def planted_frame_activations(frame_name: str, n_scenes=40, rng=None):
    """Activations encode the egg position linearly in ONE frame; other frames are nonlinear images of it."""
    rng = rng or np.random.default_rng(11)
    meta = factorial_meta(n_scenes)
    rows = []
    for i in range(len(meta["pair_id"])):
        g = fake_geometry(meta["seed"][i], meta["hazard"][i], meta["action"][i])
        rows.append(gf.frame_coordinates(gf.CellGeometry("c", g["egg"], g["eef"], g["euler"], g["actions"], (256, 256),
                                                          g["cam_pos"], g["cam_mat"], 85.0, g["eef_px"], g["egg_px"])))
    coords = {n: np.stack([r[0][n] for r in rows]) for n in gf.FRAME_NAMES}
    C = coords[frame_name]
    W = rng.normal(size=(C.shape[1], D))
    X = C @ W * 10 + rng.normal(size=(len(C), D)) * 0.05
    meta["angle"] = np.asarray([r[1]["bearing"] for r in rows])
    return X, coords, meta


def test_frame_table_ranks_the_planted_frame_first():
    X, coords, meta = planted_frame_activations("gripper")
    Xc = scene_center(X, meta["pair_id"])
    tab = gf.frame_table(Xc, coords, meta, budget=3, lam=1e-2, n_folds=5, n_boot=30, seed=0)
    r2 = {n: v["r2"]["point"] for n, v in tab["frames"].items() if v["status"] == "ok"}
    assert tab["ranking"][0]["frame"] == "gripper"
    assert r2["gripper"] > 0.95
    # world is a rotation of gripper per scene (random yaw) -> within-scene it is NOT linear in the same coords
    assert r2["world"] < r2["gripper"] - 0.1
    # world and camera are rigid transforms of each other: same readout subspace, different numbers
    sim = tab["readout_similarity"]["principal_angles_mean_sq_cos"]
    assert sim["world"]["camera"] > 0.9
    assert tab["readout_similarity"]["linear_cka"]["world"]["camera"] > 0.9
    # path lateral offsets depend on the (vertical) action only through its direction, so the DiD is flat
    assert tab["frames"]["path"]["interaction_status"] in ("action_independent_frame", "tested")


def test_frame_table_world_planted_makes_world_and_camera_equivalent():
    X, coords, meta = planted_frame_activations("world")
    Xc = scene_center(X, meta["pair_id"])
    tab = gf.frame_table(Xc, coords, meta, budget=3, lam=1e-2, n_folds=5, n_boot=30, seed=0)
    r2 = {n: v["r2"]["point"] for n, v in tab["frames"].items() if v["status"] == "ok"}
    assert r2["world"] > 0.95 and r2["camera"] > 0.9
    assert r2["gripper"] < r2["world"] - 0.1


# ----------------------------------------------------------------------------- counterfactual reference + on-manifold


def test_counterfactual_direction_recovers_hazard_axis_and_stays_on_manifold():
    X, y, meta, w = planted_linear()
    Xc = scene_center(X, meta["pair_id"])
    m = CounterfactualPairDirection().fit(Xc, y, meta)
    assert m.n_pairs == 80
    # the H0->H1 delta carries the planted direction only through the interaction cell (a=1)
    assert abs(m.direction @ w) > 0.9
    stats = {"mean": Xc.mean(0), "var": Xc.var(0) + 1e-8}
    idx = np.flatnonzero(y < 0.5)[:20]
    deltas = np.stack([m.edit_direction(Xc[i], 3.0, 1.5) for i in idx])
    om = on_manifold_scores(Xc[idx], deltas, stats)
    om_rand = on_manifold_scores(Xc[idx], random_equal_norm_edits(deltas, 0), stats)
    # an equal-norm step along the real counterfactual axis inflates Mahalanobis less than a random direction
    assert om["increase_mean"] < om_rand["increase_mean"]
    assert mahalanobis_diag(Xc[:3], stats).shape == (3,)


# ----------------------------------------------------------------------------- interaction target


def test_interaction_target_and_prediction():
    meta = factorial_meta(6)
    contact = (meta["hazard"] * meta["action"]).astype(float)
    force = contact * 10
    y = gt.interaction_target(contact, force, meta["pair_id"], meta["hazard"], meta["action"])
    assert np.all(np.isfinite(y))
    assert y[(meta["hazard"] == 1) & (meta["action"] == 1)].min() == 1.0
    assert y[(meta["hazard"] == 1) & (meta["action"] == 0)].max() == 0.0
    assert y[(meta["hazard"] == 0) & (meta["action"] == 0)].min() == 1.0
    # a scene with no DiD gets nan
    contact2 = contact.copy(); contact2[:4] = 0
    y2 = gt.interaction_target(contact2, force * 0, meta["pair_id"], meta["hazard"], meta["action"])
    assert np.all(np.isnan(y2[:4])) and np.all(np.isfinite(y2[4:]))
    # activations with only main effects cannot predict the interaction; with an interaction term they can
    rng = np.random.default_rng(5)
    meta = factorial_meta(40)
    n = len(meta["pair_id"])
    wh, wa, wi = (rng.normal(size=D) for _ in range(3))
    main_only = np.outer(meta["hazard"], wh) + np.outer(meta["action"], wa) + rng.normal(size=(n, D)) * 0.1
    with_int = main_only + np.outer(meta["hazard"] * meta["action"], wi)
    yi = gt.interaction_target((meta["hazard"] * meta["action"]).astype(float), np.zeros(n), meta["pair_id"], meta["hazard"], meta["action"])
    au_main = cv_metric(lambda: LinearRidge(lam=1.0), scene_center(main_only, meta["pair_id"]), yi, meta, "binary")
    au_int = cv_metric(lambda: LinearRidge(lam=1.0), scene_center(with_int, meta["pair_id"]), yi, meta, "binary")
    assert au_int > 0.9
    assert au_main < 0.65


def test_rank_sweep_finds_planted_rank():
    X, y, meta, _ = planted_lowrank(rank=3)
    Xc = scene_center(X, meta["pair_id"])
    args = type("A", (), {"lam": 1e-2, "n_folds": 5, "split_seed": 0})()
    sweep = gt.rank_sweep("continuous", Xc, y, meta, args)
    assert sweep["rank_needed_for_gate"] is not None and sweep["rank_needed_for_gate"] <= 3
    assert [c["rank"] for c in sweep["curve"]][:3] == [1, 2, 3]


# ----------------------------------------------------------------------------- end-to-end on a fake dump


def write_fake_dump(tmp: Path, X: np.ndarray, y_force: np.ndarray, meta: dict[str, np.ndarray], families=None):
    dump = tmp / "dump"
    stim = tmp / "stim"
    (dump / "activations").mkdir(parents=True)
    (stim / "cells").mkdir(parents=True)
    n = len(y_force)
    index = []
    rows = []
    for i in range(n):
        cell_id = f"{meta['pair_id'][i]}__h{meta['hazard'][i]}a{meta['action'][i]}"
        index.append(
            {
                "pair_id": str(meta["pair_id"][i]),
                "hazard": int(meta["hazard"][i]),
                "candidate_action": int(meta["action"][i]),
                "cell_id": cell_id,
                "seed": int(meta["seed"][i]),
            }
        )
        art = f"cells/{cell_id}.npz"
        geo = fake_geometry(meta["seed"][i], meta["hazard"][i], meta["action"][i])
        np.savez(
            stim / art,
            model_actions=geo["actions"].astype(np.float32),
            context_proprios=geo["proprios"].astype(np.float32),
            egg_initial_position=geo["egg"],
            context_frames=np.zeros((1, 256, 256, 3), dtype=np.uint8),
        )
        (stim / "masks").mkdir(exist_ok=True)
        np.savez(stim / "masks" / f"{cell_id}.npz", cam_pos=geo["cam_pos"], cam_mat=geo["cam_mat"], fovy=85.0,
                 eef_px=geo["eef_px"], egg_px=geo["egg_px"])
        contact = int(meta["hazard"][i] == 1 and meta["action"][i] == 1)
        rows.append(
            {
                "cell_id": cell_id,
                "pair_id": str(meta["pair_id"][i]),
                "artifact": art,
                "hazard": int(meta["hazard"][i]),
                "candidate_action": int(meta["action"][i]),
                "egg_robot_contact_count": 8 * contact,
                "max_normal_force_n": float(y_force[i]),
                "gripper_to_target_approach_m": 0.01 if meta["hazard"][i] else 0.15,
                "target_centroid_xy": [110.0, 112.0],
                "target_visible_pixels": 50,
                "risk_family": (families[i] if families is not None else "gentle_contact"),
            }
        )
    (dump / "activations" / "index.json").write_text(json.dumps(index))
    (stim / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    pairs = [
        {"pair_id": p, "control_pose_selection": {"angle_rad": float(np.pi)}}
        for p in np.unique(meta["pair_id"])
    ]
    (stim / "summary.json").write_text(json.dumps({"pairs": pairs}))
    # acts: [n, T=2, N=3, d]; put the signal at the last step, mean over tokens preserves it
    acts = np.zeros((n, 2, 3, D), dtype=np.float16)
    acts[:, 1] = X[:, None, :].astype(np.float16)
    acts[:, 0] = np.random.default_rng(0).normal(size=(n, 3, D)).astype(np.float16)
    np.savez_compressed(dump / "activations" / "L03__resid_post.npz", acts=acts)
    return dump, stim


def test_end_to_end_tournament_on_fake_dump(tmp_path: Path):
    X, y, meta, _ = planted_linear(n_scenes=40, main_effects=True)
    y = np.clip(y, 0, None) * 6.0  # force-like label, positive
    dump, stim = write_fake_dump(tmp_path, X, y, meta)
    seeds = sorted(set(meta["seed"].tolist()))
    (tmp_path / "disc.txt").write_text("\n".join(str(s) for s in seeds[:28]))
    (tmp_path / "conf.txt").write_text("\n".join(str(s) for s in seeds[28:]))
    report = gt.main(
        [
            "--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "out"),
            "--discovery-seeds", str(tmp_path / "disc.txt"), "--confirmation-seeds", str(tmp_path / "conf.txt"),
            "--budget", "3", "--n-boot", "50", "--export-edits", "--targets", "contact", "force", "interaction",
        ]
    )
    assert (tmp_path / "out" / "tournament.json").exists()
    site = report["sites"][0]
    assert site["site_id"] == "L03__resid_post"
    lin = site["models"]["linear_ridge"]
    assert lin["contact"]["confirmation"]["point"] >= 0.75
    assert lin["contact"]["gate"]["metric_pass"] is True
    assert lin["force"]["confirmation"]["point"] >= 0.30
    assert lin["contact"]["transport"]["status"] == "task_local"
    assert lin["contact"]["stability"]["mean_similarity"] > 0.8
    assert lin["contact"]["gate"]["stability_pass"] is True
    # within-scene DiD sign agreement on confirmation scenes
    ws = lin["force"]["confirmation_within_scene"]
    assert ws["n_scenes"] == 12 and ws["sign_agreement"] >= 0.8
    # edit artifacts exist and carry one delta per confirmation cell
    edit = Path(lin["contact"]["edit_artifact"])
    assert edit.exists()
    with np.load(edit) as z:
        assert z["delta"].shape == (48, D)
        assert len(z["cell_ids"]) == 48
    inject = json.loads((edit.parent / (edit.name[:-4] + ".json")).read_text())
    assert inject["site_id"] == "L03__resid_post" and inject["token_pool"] == "mean" and inject["step"] == -1
    # frames stage ran and the circular angle came from the path bearing
    assert report["frames_stage"] == "ran" and report["angle_source"] == "path_bearing"
    assert set(site["frames"]["frames"]) == set(gf.FRAME_NAMES)
    assert site["frames"]["reprojection_error_px"]["max"] < 1e-3
    # interaction target evaluated and reported in the best-model table
    assert "interaction" in site["best_model_by_target"]
    assert site["models"]["linear_ridge"]["interaction"]["confirmation"]["point"] > 0.9
    # counterfactual reference entrant and edit comparison
    comp = site["edit_comparison"]["contact"]
    assert "counterfactual_pair_direction" in comp and "linear_ridge" in comp
    assert -1.0 <= comp["linear_ridge"]["cosine_to_counterfactual"] <= 1.0
    assert "on_manifold_random_equal_norm" in site["models"]["linear_ridge"]["contact"]
    assert "rank_sweep" in site["models"]["lowrank_subspace"]["contact"]
    assert len(report["nuisance_columns"]) == 7
    # every model produced a result or an explicit error, never silently missing
    for name, entry in site["models"].items():
        assert "contact" in entry and ("gate" in entry["contact"] or "error" in entry["contact"]), name


def test_dict_index_and_padded_token_index(tmp_path: Path):
    """Localization-dump schema: index.json is a dict with `cells` (rows) and acts carry a padded token_index."""
    X, y, meta, _ = planted_linear(n_scenes=12, main_effects=True)
    dump, stim = write_fake_dump(tmp_path, X, np.abs(y) * 6, meta)
    cells = json.loads((dump / "activations" / "index.json").read_text())
    for i, c in enumerate(cells):
        c["row"] = i
    (dump / "activations" / "index.json").write_text(json.dumps({"cells": cells[::-1], "n_steps": 2, "sites": {}, "group_frame_offset": 0}))
    with np.load(dump / "activations" / "L03__resid_post.npz") as z:
        acts = np.asarray(z["acts"])
    n, T, N, d = acts.shape
    # pad with a garbage token that must be ignored by the masked mean
    acts2 = np.concatenate([acts, np.full((n, T, 1, d), 1e3, np.float16)], axis=2)
    tix = np.concatenate([np.tile(np.arange(N), (n, T, 1)), np.full((n, T, 1), -1)], axis=2)
    np.savez_compressed(dump / "activations" / "L03__resid_post.npz", acts=acts2, token_index=tix)
    idx = gt.load_index(dump)
    assert [c["row"] for c in idx] == list(range(n))
    feats = gt.load_site_features(dump, "L03__resid_post", -1, "mean")
    assert np.allclose(feats, acts[:, -1].astype(np.float32).mean(axis=1), atol=1e-2)
    report = gt.main(["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "o"), "--n-boot", "20",
                      "--models", "linear_ridge", "--targets", "force", "--no-frames"])
    assert report["dump_index_meta"]["n_steps"] == 2
    assert report["sites"][0]["models"]["linear_ridge"]["force"]["discovery_cv"]["point"] > 0.3


def test_calibration_seeds_are_refused(tmp_path: Path):
    X, y, meta, _ = planted_linear(n_scenes=6)
    meta["seed"] = np.where(meta["seed"] == 200, 101, meta["seed"])
    meta["pair_id"] = np.where(meta["seed"] == 101, "contact_seed000101", meta["pair_id"])
    dump, stim = write_fake_dump(tmp_path, X, np.abs(y), meta)
    with pytest.raises(SystemExit, match="calibration"):
        gt.main(["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "o"), "--n-boot", "5"])


def test_transport_reported_when_two_families(tmp_path: Path):
    X, y, meta, _ = planted_linear(n_scenes=30)
    fam = np.where(meta["seed"] % 2 == 0, "gentle_contact", "route_conflict")
    dump, stim = write_fake_dump(tmp_path, X, np.abs(y) * 6, meta, families=fam)
    report = gt.main(
        ["--dump", str(dump), "--stimulus", str(stim), "--out", str(tmp_path / "o"), "--n-boot", "20",
         "--models", "linear_ridge", "--targets", "force"]
    )
    tr = report["sites"][0]["models"]["linear_ridge"]["force"]["transport"]
    assert tr["status"] == "tested"
    assert set(tr["by_source_family"]) == {"gentle_contact", "route_conflict"}


def test_write_edit_artifact_format(tmp_path: Path):
    p = write_edit_artifact(
        tmp_path / "e", model_name="m", site_id="s", step=-1, token_pool="mean",
        cell_ids=["a", "b"], deltas=np.ones((2, 4)), norm=0.5, target=1.0, target_name="contact",
    )
    with np.load(p) as z:
        assert z["delta"].dtype == np.float32 and float(z["norm"]) == 0.5
    meta = json.loads((p.parent / (p.name[:-4] + ".json")).read_text())
    assert meta["n_cells"] == 2 and "inject" in meta
