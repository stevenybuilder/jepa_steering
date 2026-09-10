"""CPU tests for geometry_localize: a planted rotating-direction interaction code (direction = f(bearing))
that the global-mean LOSO projection misses and the local / kNN / RSA tests catch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

import geometry_localize as gl  # noqa: E402
from cgs_stats import loso_projection_scores  # noqa: E402
from token_groups import load_cell_groups, union  # noqa: E402

D = 64
N_SCENES = 12


def orthonormal(rng, k):
    return np.linalg.qr(rng.normal(size=(D, k)))[0].T


def look_at(cam_pos, target):
    z = cam_pos - target
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def build_stimulus(tmp: Path, n_scenes: int, seed: int = 0) -> tuple[Path, list[dict], dict, np.ndarray]:
    """Write cells + masks + manifest; return (stimulus dir, index cells, manifest, bearings)."""
    rng = np.random.default_rng(seed)
    stim = tmp / "stim"
    (stim / "cells").mkdir(parents=True)
    (stim / "masks").mkdir()
    cells, rows, bearings = [], [], []
    row = 0
    for s in range(n_scenes):
        seed_id = 300 + s
        pid = f"contact_seed{seed_id:06d}"
        b = -np.pi + 2 * np.pi * (s + 0.5) / n_scenes  # bearings cover the full circle
        bearings.append(b)
        h1 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-1.8, -1.4), 0.94])
        # eef 20 cm above the H1 spot and a close camera so egg / gripper patches do not overlap (real
        # scenes: ~24 px apart with gripper groups of 5-15 patches)
        eef = h1 + np.array([0.0, 0.0, 0.20])
        euler = np.array([np.pi, 0.0, rng.uniform(-np.pi, np.pi)])
        cam_pos = h1 + np.array([0.4, 0.55, 0.5])
        cam_mat = look_at(cam_pos, h1)
        for h in (0, 1):
            egg = h1 if h else h1 + 0.15 * np.array([np.cos(b), np.sin(b), 0.0])
            for a in (0, 1):
                cid = f"{pid}__h{h}a{a}"
                actions = np.zeros((3, 7))
                actions[:, 2] = -0.006 * (1 + a)
                pro = np.stack([np.concatenate([eef, euler, [0.0]])] * 2)
                art = f"cells/{cid}.npz"
                np.savez(stim / art, egg_initial_position=egg, context_proprios=pro.astype(np.float32),
                         model_actions=actions.astype(np.float32), context_frames=np.zeros((2, 256, 256, 3), np.uint8))
                # masks: egg blob around its projected pixel, robot blob around the eef pixel
                from geometry_frames import project_to_pixels
                egg_px = project_to_pixels(egg, cam_pos, cam_mat, 85.0, (256, 256))
                eef_px = project_to_pixels(eef, cam_pos, cam_mat, 85.0, (256, 256))
                egg_mask = np.zeros((4, 256, 256), bool)
                robot_mask = np.zeros((4, 256, 256), bool)
                yy, xx = np.mgrid[:256, :256]
                for f in range(4):
                    egg_mask[f] = (xx - egg_px[0]) ** 2 + (yy - egg_px[1]) ** 2 < 6**2
                    robot_mask[f] = (xx - eef_px[0]) ** 2 + (yy - eef_px[1]) ** 2 < 12**2
                np.savez(stim / "masks" / f"{cid}.npz", egg_mask=egg_mask, robot_mask=robot_mask,
                         eef_px=np.tile(eef_px, (4, 1)).astype(np.float32), egg_px=np.tile(egg_px, (4, 1)).astype(np.float32),
                         cam_pos=cam_pos, cam_mat=cam_mat, fovy=85.0, frame_index_meaning=np.array(["context", "future1", "future2", "future3"]))
                contact = int(h == 1 and a == 1)
                rows.append({"cell_id": cid, "pair_id": pid, "artifact": art, "hazard": h, "candidate_action": a,
                             "egg_robot_contact_count": 8 * contact, "max_normal_force_n": 15.0 * contact,
                             "gripper_to_target_approach_m": 0.01 if h else 0.15, "risk_family": "gentle_contact"})
                cells.append({"row": row, "pair_id": pid, "seed": seed_id, "hazard": h, "candidate_action": a, "cell_id": cid,
                              "artifact": str((stim / art).resolve())})
                row += 1
    (stim / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return stim, cells, {r["cell_id"]: r for r in rows}, np.asarray(bearings)


def plant_code(cells, bearings, rotating: bool, seed: int = 1, noise: float = 0.05):
    """Cell vectors: mean_s + h*H_s + a*A_s + (2h-1)(2a-1)/4 * I_s + noise, with I_s rotating or fixed."""
    rng = np.random.default_rng(seed)
    wh, wa, p, q, u, v, m = orthonormal(rng, 7)
    n = len(cells)
    X = np.zeros((n, D))
    for c in cells:
        s = int(c["seed"]) - 300
        b = bearings[s]
        H = 2.0 * wh + 1.5 * (np.cos(b) * p + np.sin(b) * q)  # hazard main effect carries the geometry
        A = 2.0 * wa
        I = 4.0 * (np.cos(b) * u + np.sin(b) * v) if rotating else 4.0 * u
        h, a = c["hazard"], c["candidate_action"]
        X[c["row"]] = 3.0 * m * (s / 10.0) + h * H + a * A + (2 * h - 1) * (2 * a - 1) / 4.0 * I + rng.normal(size=D) * noise
    return X, (u, v)


def build_dump(tmp: Path, stim: Path, cells, X, sites=("L00.attn_out", "L07.attn_out")) -> Path:
    dump = tmp / "dump"
    (dump / "activations").mkdir(parents=True)
    n_steps = 3
    # token sets exactly as the dump would: union over the quartet of egg | gripper | corridor per step
    groups = {c["cell_id"]: load_cell_groups(stim, {"cell_id": c["cell_id"]}) for c in cells}
    by_pair = {}
    for c in cells:
        by_pair.setdefault(c["pair_id"], []).append(c)
    k = 0
    tok = {}
    for pid, mem in by_pair.items():
        for step in range(n_steps):
            t = union(*(groups[c["cell_id"]].group(step, g) for c in mem for g in ("egg", "gripper", "corridor")))
            tok[(pid, step)] = t
            k = max(k, len(t))
    n = len(cells)
    token_index = np.full((n, n_steps, k), -1, dtype=np.int64)
    acts = np.zeros((n, n_steps, k, D), np.float16)
    rng = np.random.default_rng(0)
    for c in cells:
        for step in range(n_steps):
            t = tok[(c["pair_id"], step)]
            token_index[c["row"], step, : len(t)] = t
            acts[c["row"], step, : len(t)] = (X[c["row"]][None, :] + rng.normal(size=(len(t), D)) * 0.01).astype(np.float16)
    for s in sites:
        np.savez_compressed(dump / "activations" / f"{s}.npz", acts=acts, token_index=token_index)
    (dump / "activations" / "index.json").write_text(json.dumps({"cells": cells, "n_steps": n_steps, "group_frame_offset": 0, "dump_groups": ["egg", "gripper", "corridor"]}))
    return dump


def scene_vectors(X, cells):
    Q = gl.quartets(cells)
    return np.stack([gl.scene_effects(X, Q[p])["interaction"] for p in sorted(Q)]), sorted(Q)


# ----------------------------------------------------------------------------- unit pieces


def test_kernel_ridge_learns_xor_and_twonn_recovers_dimension():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    m = gl.KernelRidgeRBF(lam=0.1).fit(X[:200], y[:200], {})
    from geometry_models import auroc
    assert auroc(m.score(X[200:]), y[200:]) > 0.95
    # 2-d manifold embedded in D dims
    basis = orthonormal(rng, 2)
    Z = rng.normal(size=(500, 2)) @ basis + rng.normal(size=(500, D)) * 1e-3
    assert 1.5 < gl.twonn_dimension(Z) < 2.8
    assert 1.5 < gl.local_pca_participation_ratio(Z, k=15) < 2.5
    Z8 = rng.normal(size=(500, 8)) @ orthonormal(rng, 8)
    assert gl.twonn_dimension(Z8) > gl.twonn_dimension(Z) + 3
    # near-duplicate rows (four copies of each point) collapse TwoNN; dedupe restores it
    Zdup = np.repeat(Z, 4, axis=0) + rng.normal(size=(2000, D)) * 1e-5
    assert not (1.5 < gl.twonn_dimension(Zdup) < 2.8)  # duplicates break the estimator (here: reads the noise)
    Zd, dropped = gl.dedupe_near_duplicates(Zdup)
    assert 0.7 < dropped < 0.8
    assert 1.5 < gl.twonn_dimension(Zd) < 2.8


def test_residualization_leaves_interaction_only():
    rng = np.random.default_rng(3)
    cells = [{"row": i, "pair_id": f"p{i // 4}", "seed": 300 + i // 4, "hazard": (i // 2) % 2, "candidate_action": i % 2, "cell_id": f"c{i}"} for i in range(8)]
    X, _ = plant_code(cells, np.array([0.3, 1.1]), rotating=False, noise=0.0)
    Q = gl.quartets(cells)
    strict = gl.residualize(X, Q, "strict")
    keep = gl.residualize(X, Q, "keep_hazard")
    I = gl.scene_effects(X, Q["p0"])["interaction"]
    assert np.allclose(strict[3], I / 4) and np.allclose(strict[2], -I / 4)
    H = gl.scene_effects(X, Q["p0"])["hazard"]
    assert np.allclose(keep[3], H / 2 + I / 4) and np.allclose(keep[0], -H / 2 + I / 4)


def test_rotating_code_missed_by_global_caught_by_local_and_rsa(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, N_SCENES)
    Q = gl.quartets(cells)
    cov = gl.scene_covariates(stim, cells, manifest, Q)
    # the stimulus bearing recovered from the path frame matches the planted one
    assert np.max(np.abs(np.angle(np.exp(1j * (cov["bearing"] - bearings))))) < 1e-6
    for rotating in (True, False):
        X, _ = plant_code(cells, bearings, rotating=rotating)
        I, order = scene_vectors(X, cells)
        sel = np.asarray([cov["pair_ids"].index(p) for p in order])
        lvg = gl.local_vs_global_projection(I, cov["covariates"][sel], k=4, n_boot=100, seed=0)
        dg = gl.rsa_direction_geometry(I, cov["bearing"][sel], cov["covariates"][sel], n_perm=299, seed=0)
        if rotating:
            assert lvg["global"]["t"] < 2.0  # no positive global signal: the linear map misses it
            assert lvg["local"]["t"] > 3.0
            assert lvg["local_minus_global"]["ci_low"] > 0.0
            assert gl.curved_code(lvg)
            assert dg["rsa_bearing"]["spearman"] > 0.5 and dg["rsa_bearing"]["p_perm"] < 0.05
            assert dg["circular_fit"]["advantage"] > 0.5
        else:
            assert lvg["global"]["t"] > 5.0
            assert 0.8 < lvg["ratio_local_over_global"]["point"] < 1.25
            assert not gl.curved_code(lvg)
            assert dg["circular_fit"]["advantage"] < 0.1


def test_readouts_nonlinear_wins_only_for_rotating_code(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, N_SCENES)
    Q = gl.quartets(cells)
    labels = gl.interaction_labels(cells, manifest, Q)
    n = len(cells)
    meta = {"pair_id": np.asarray([c["pair_id"] for c in cells]), "hazard": np.asarray([c["hazard"] for c in cells]),
            "action": np.asarray([c["candidate_action"] for c in cells])}
    for rotating in (True, False):
        X, _ = plant_code(cells, bearings, rotating=rotating)
        R = gl.residualize(X, Q, "keep_hazard")
        rc = gl.readout_comparison(R, labels["interaction"], meta, "binary", budget=4, lam=1.0, n_boot=100, seed=0)
        assert rc["status"] == "ok" and rc["n_scenes"] == N_SCENES
        lin = rc["readouts"]["linear_ridge"]["point"]
        best = rc["readouts"][rc["best_nonlinear"]]["point"]
        if rotating:
            assert lin < 0.75, lin
            assert best > 0.9, (rc["best_nonlinear"], best)
            assert rc["nonlinear_wins"]
        else:
            assert lin > 0.9
            assert rc["nonlinear_advantage"] < 0.1


def test_end_to_end_map_and_ranked_tables(tmp_path: Path):
    stim, cells, manifest, bearings = build_stimulus(tmp_path, N_SCENES)
    X_rot, _ = plant_code(cells, bearings, rotating=True)
    X_fix, _ = plant_code(cells, bearings, rotating=False, seed=2)
    dump = build_dump(tmp_path, stim, cells, X_rot, sites=("L07.attn_out",))
    # second site with a fixed code, written into the same dump
    build_dump(tmp_path / "tmp2", stim, cells, X_fix, sites=("L00.attn_out",))
    (dump / "activations" / "L00.attn_out.npz").write_bytes((tmp_path / "tmp2" / "dump" / "activations" / "L00.attn_out.npz").read_bytes())
    linear_map = {"maps_by_step": {"step0": {"top10_gripper_corridor": [
        {"site_id": "L00.attn_out", "step": 0, "group": "gripper_corridor", "t": 6.1, "p_maxt_fwer": 0.01},
        {"site_id": "L07.attn_out", "step": 0, "group": "gripper_corridor", "t": 0.4, "p_maxt_fwer": 0.9},
    ]}}}
    (tmp_path / "lin.json").write_text(json.dumps(linear_map))
    report = gl.main(["--dump", str(dump), "--stimulus", str(stim), "--linear-map", str(tmp_path / "lin.json"),
                      "--out", str(tmp_path / "out"), "--steps", "0", "--groups", "gripper_corridor", "egg",
                      "--n-boot", "100", "--n-perm", "199"])
    assert (tmp_path / "out" / "nonlinear_map.json").exists()
    tab = report["ranked_tables"]["step0|gripper_corridor"]
    by = {r["site_id"]: r for r in tab}
    assert by["L07.attn_out"]["linear_map_t"] == 0.4 and by["L00.attn_out"]["linear_map_t"] == 6.1
    assert tab[0]["site_id"] == "L07.attn_out"  # rotating code ranks first by nonlinear advantage
    assert by["L07.attn_out"]["curved_code_flag"] is True
    assert by["L00.attn_out"]["curved_code_flag"] is False
    assert any(r["site_id"] == "L07.attn_out" for r in report["curved_or_nonlinear_but_low_linear_t"])
    assert report["focus_site_step0"][0]["site_id"] == "L07.attn_out"
    assert report["scenes_with_hazard2"] == 0
    assert report["entries"][0]["readouts"]["null_contrast"]["status"] == "hazard2_cells_not_in_dump"
    assert "pooled_tokens" in report["entries"][0]["density"]
