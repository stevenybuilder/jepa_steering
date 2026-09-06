#!/usr/bin/env python3
"""Candidate coordinate frames for the hazard variable ("Triangulate" stage of CGS).

Othello lesson: a failed probe can mean the *target coordinate system* is
wrong, not that the variable is absent. So before judging any chart we express
the hazard (egg) position in several candidate frames built from the raw
per-cell records and ask which frame the site's activations read out most
simply (equal-budget ridge, held-out by scene). Frames that are rotations of
one another (world vs camera) give different numbers but the same geometry;
the rotation-invariant comparison (principal angles / linear CKA between the
readout subspaces) is reported so that case is detectable.

Consumed fields
---------------
cell npz (``<stimulus>/<artifact>``):
    ``egg_initial_position`` [3]        world xyz of the egg
    ``context_proprios`` [T, 7]         row -1: [:3] eef xyz, [3:6] eef euler (extrinsic xyz), [6] gripper
    ``model_actions`` [H, 7]            action chunk, [:, :3] treated as xyz deltas (direction only)
    ``context_frames`` [T, H, W, 3]     only its H, W are used (default 256 x 256 if absent)
mask npz (``<stimulus>/masks/<cell_id>.npz``), optional:
    ``cam_pos`` [3], ``cam_mat`` [3, 3], ``fovy`` scalar (deg), ``eef_px`` [2], ``egg_px`` [4, 2]
    Without a mask the camera / image frames are skipped and reported as such.

Frames (name -> coordinates)
----------------------------
world             egg xyz
camera            cam_mat.T @ (egg - cam_pos)
image             pixel (u, v) of the egg: mean of ``egg_px`` if present, else projected
gripper           R_eef.T @ (egg - eef)
path              along-path distance, signed lateral offsets (u1, u2) w.r.t. the descent line
hazard_under_path min distance from the egg to the candidate path segment (scalar)
extras            ``bearing`` (atan2 of the lateral offsets; the circular variable), ``lateral_dist``

The descent line starts at the eef at t0 and follows the cumulative xyz of the
action chunk (falls back to straight down); the segment ends where it reaches
the egg's height plane (hover offset is 0.075 m above the H1 egg by construction).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geometry_models import cluster_bootstrap, r_squared, ridge_fit, ridge_predict, scene_center, scene_folds, subspace_similarity

HOVER_OFFSET_M = 0.075
MASK_FIELDS = ("cam_pos", "cam_mat", "fovy", "eef_px", "egg_px")
CELL_FIELDS = ("egg_initial_position", "context_proprios", "model_actions")
FRAME_NAMES = ("world", "camera", "image", "gripper", "path", "hazard_under_path")


@dataclass
class CellGeometry:
    cell_id: str
    egg: np.ndarray
    eef_pos: np.ndarray
    eef_euler: np.ndarray
    actions: np.ndarray
    image_hw: tuple[int, int]
    cam_pos: np.ndarray | None = None
    cam_mat: np.ndarray | None = None
    fovy: float | None = None
    eef_px: np.ndarray | None = None
    egg_px: np.ndarray | None = None


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Extrinsic x-y-z Euler angles (robosuite ``mat2euler`` convention) -> rotation matrix R = Rz Ry Rx."""
    a, b, c = (float(v) for v in np.asarray(euler, dtype=np.float64)[:3])
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def project_to_pixels(p: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, fovy_deg: float, image_hw: tuple[int, int]) -> np.ndarray:
    """MuJoCo camera model: looks down -z of ``cam_mat`` with +y up. Returns (u, v) with v downwards."""
    h, w = image_hw
    pc = np.asarray(cam_mat, dtype=np.float64).T @ (np.asarray(p, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    depth = -pc[2]
    if depth <= 1e-9:
        return np.array([np.nan, np.nan])
    f = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    u = w / 2.0 + f * pc[0] / depth
    v = h / 2.0 - f * pc[1] / depth
    return np.array([u, v])


def load_cell_geometry(stimulus: Path, row: dict[str, Any]) -> CellGeometry:
    with np.load(stimulus / row["artifact"]) as z:
        missing = [k for k in CELL_FIELDS if k not in z]
        if missing:
            raise KeyError(f"{row['cell_id']}: cell npz missing {missing}")
        egg = np.asarray(z["egg_initial_position"], dtype=np.float64)[:3]
        pro = np.asarray(z["context_proprios"], dtype=np.float64)[-1]
        actions = np.asarray(z["model_actions"], dtype=np.float64)
        hw = tuple(int(v) for v in z["context_frames"].shape[1:3]) if "context_frames" in z else (256, 256)
    g = CellGeometry(row["cell_id"], egg, pro[:3], pro[3:6], actions, hw)
    mask_path = stimulus / "masks" / f"{row['cell_id']}.npz"
    if mask_path.exists():
        with np.load(mask_path) as m:
            if all(k in m for k in ("cam_pos", "cam_mat", "fovy")):
                g.cam_pos = np.asarray(m["cam_pos"], dtype=np.float64)
                g.cam_mat = np.asarray(m["cam_mat"], dtype=np.float64).reshape(3, 3)
                g.fovy = float(np.asarray(m["fovy"]).ravel()[0])
            if "eef_px" in m:
                g.eef_px = np.asarray(m["eef_px"], dtype=np.float64).ravel()[:2]
            if "egg_px" in m:
                g.egg_px = np.asarray(m["egg_px"], dtype=np.float64).reshape(-1, 2)
    return g


def path_frame(g: CellGeometry) -> dict[str, Any]:
    start = g.eef_pos
    disp = g.actions[:, :3].sum(axis=0) if g.actions.ndim == 2 and g.actions.shape[1] >= 3 else np.zeros(3)
    if np.linalg.norm(disp) < 1e-9:
        disp = np.array([0.0, 0.0, -1.0])
    d = disp / np.linalg.norm(disp)
    # segment reaches the egg's height plane when descending, else a nominal 0.1 m
    length = (start[2] - g.egg[2]) / (-d[2]) if d[2] < -1e-6 else 0.1
    length = float(max(length, 1e-3))
    up = np.array([0.0, 0.0, 1.0])
    u1 = np.cross(d, up)
    if np.linalg.norm(u1) < 1e-6:  # vertical path: lateral basis = world x, y
        u1 = np.array([1.0, 0.0, 0.0])
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u1, d)  # for a vertical descent this gives (u1, u2) = (world x, world y)
    u2 /= np.linalg.norm(u2)
    rel = g.egg - start
    along = float(rel @ d)
    lat = rel - along * d
    l1, l2 = float(lat @ u1), float(lat @ u2)
    t = float(np.clip(along, 0.0, length))
    under = float(np.linalg.norm(rel - t * d))
    return {
        "path": np.array([along, l1, l2]),
        "hazard_under_path": np.array([under]),
        "bearing": float(np.arctan2(l2, l1)),
        "lateral_dist": float(np.hypot(l1, l2)),
        "segment_length": length,
        "direction": d,
    }


def frame_coordinates(g: CellGeometry) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
    """Return (frames, extras, diagnostics) for one cell."""
    frames: dict[str, np.ndarray] = {"world": g.egg.copy()}
    diag: dict[str, Any] = {}
    if g.cam_mat is not None:
        frames["camera"] = g.cam_mat.T @ (g.egg - g.cam_pos)
        proj = project_to_pixels(g.egg, g.cam_pos, g.cam_mat, g.fovy, g.image_hw)
        if g.egg_px is not None:
            frames["image"] = g.egg_px.mean(axis=0)
            diag["reprojection_error_px"] = float(np.linalg.norm(proj - frames["image"])) if np.all(np.isfinite(proj)) else float("nan")
        else:
            frames["image"] = proj
    elif g.egg_px is not None:
        frames["image"] = g.egg_px.mean(axis=0)
    R = euler_xyz_to_matrix(g.eef_euler)
    frames["gripper"] = R.T @ (g.egg - g.eef_pos)
    pf = path_frame(g)
    frames["path"] = pf["path"]
    frames["hazard_under_path"] = pf["hazard_under_path"]
    extras = {"bearing": pf["bearing"], "lateral_dist": pf["lateral_dist"], "segment_length": pf["segment_length"]}
    return frames, extras, diag


def collect_frames(stimulus: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frames for every cell (in the given order). Frames missing for any cell are dropped with a note."""
    per_cell = []
    for row in rows:
        g = load_cell_geometry(stimulus, row)
        per_cell.append(frame_coordinates(g))
    names = [n for n in FRAME_NAMES if all(n in fc[0] for fc in per_cell)]
    dropped = [n for n in FRAME_NAMES if n not in names]
    coords = {n: np.stack([fc[0][n] for fc in per_cell]) for n in names}
    extras = {k: np.asarray([fc[1][k] for fc in per_cell]) for k in ("bearing", "lateral_dist", "segment_length")}
    reproj = [fc[2].get("reprojection_error_px") for fc in per_cell if "reprojection_error_px" in fc[2]]
    return {
        "coords": coords,
        "extras": extras,
        "dropped_frames": dropped,
        "reprojection_error_px": {"median": float(np.nanmedian(reproj)), "max": float(np.nanmax(reproj))} if reproj else None,
    }


# --------------------------------------------------------------------------- #
# Which frame is linearly simplest for the site?
# --------------------------------------------------------------------------- #


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(0)
    B = B - B.mean(0)
    num = np.linalg.norm(A.T @ B, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(num / den) if den > 0 else float("nan")


def scene_did_rows(V: np.ndarray, meta: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-scene interaction vectors (H1A1 - H1A0 - H0A1 + H0A0). Returns (rows, pair_ids)."""
    pair, hz, ac = meta["pair_id"], meta["hazard"].astype(int), meta["action"].astype(int)
    rows, ids = [], []
    for p in np.unique(pair):
        idx = np.flatnonzero(pair == p)
        cell = {(int(hz[i]), int(ac[i])): i for i in idx}
        if all(k in cell for k in ((0, 0), (0, 1), (1, 0), (1, 1))):
            rows.append(V[cell[(1, 1)]] - V[cell[(1, 0)]] - V[cell[(0, 1)]] + V[cell[(0, 0)]])
            ids.append(p)
    return (np.asarray(rows), np.asarray(ids)) if rows else (np.zeros((0, V.shape[1])), np.asarray([]))


def frame_table(
    X: np.ndarray,
    coords: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    *,
    budget: int,
    lam: float,
    n_folds: int,
    n_boot: int,
    seed: int,
    scene_centered: bool = True,
) -> dict[str, Any]:
    """Equal-budget ridge readout from site activations to each frame's coordinates, held out by scene.

    ``X`` is expected already scene-centred when ``scene_centered``; the frame
    coordinates are centred the same way so both sides describe within-scene
    structure. Returns per-frame held-out R^2 (mean over coordinates, with a
    scene-clustered CI), per-coordinate R^2, the interaction (DiD) variant, and
    the rotation-invariant readout-subspace comparison between frames.
    """
    pair = meta["pair_id"]
    out: dict[str, Any] = {"frames": {}, "budget": budget}
    readouts: dict[str, np.ndarray] = {}
    for name, C in coords.items():
        C = np.asarray(C, dtype=np.float64)[:, :budget]
        Cc = scene_center(C, pair) if scene_centered else C - C.mean(0)
        # drop coordinates that do not vary (e.g. a constant height): their R^2 is meaningless
        scale = float(np.abs(Cc).max()) if Cc.size else 0.0
        live = np.flatnonzero(Cc.std(axis=0) > 1e-6 * max(scale, 1e-12))
        if len(live) == 0:
            out["frames"][name] = {"status": "constant_within_scene", "r2": {"point": float("nan")}}
            continue
        Cc = Cc[:, live]
        pred = np.full_like(Cc, np.nan)
        for held in scene_folds(pair, n_folds, seed):
            train = np.setdiff1d(np.arange(len(pair)), held)
            for j in range(Cc.shape[1]):
                w, b = ridge_fit(X[train], Cc[train, j], lam)
                pred[held, j] = ridge_predict(X[held], w, b)
        ok = np.all(np.isfinite(pred), axis=1)
        per_dim = [r_squared(pred[ok, j], Cc[ok, j]) for j in range(Cc.shape[1])]

        def mean_r2(idx, pred=pred, Cc=Cc, ok=ok):
            ii = np.flatnonzero(ok)[idx]
            vals = [r_squared(pred[ii, j], Cc[ii, j]) for j in range(Cc.shape[1])]
            return float(np.nanmean(vals))

        boot = cluster_bootstrap(mean_r2, pair[ok], n_boot=n_boot, seed=seed)
        W = np.stack([ridge_fit(X, Cc[:, j], lam)[0] for j in range(Cc.shape[1])])
        readouts[name] = W
        # interaction variant: do per-scene DiD activation vectors predict the DiD of the coordinate?
        Xd, ids = scene_did_rows(X, meta)
        Cd, _ = scene_did_rows(Cc, meta)
        if len(ids) >= 4 and np.any(np.std(Cd, axis=0) > 1e-9):
            dp = np.full_like(Cd, np.nan)
            for held in scene_folds(ids, min(n_folds, len(ids)), seed):
                train = np.setdiff1d(np.arange(len(ids)), held)
                for j in range(Cd.shape[1]):
                    w, b = ridge_fit(Xd[train], Cd[train, j], lam)
                    dp[held, j] = ridge_predict(Xd[held], w, b)
            did_r2 = float(np.nanmean([r_squared(dp[:, j], Cd[:, j]) for j in range(Cd.shape[1]) if np.std(Cd[:, j]) > 1e-9]))
            did_status = "tested"
        else:
            did_r2, did_status = float("nan"), "action_independent_frame" if len(ids) >= 4 else "too_few_scenes"
        out["frames"][name] = {
            "status": "ok",
            "n_coords": int(Cc.shape[1]),
            "coords_used": [int(j) for j in live],
            "r2": boot,
            "r2_per_coord": per_dim,
            "interaction_did_r2": did_r2,
            "interaction_status": did_status,
        }
    ranked = sorted(
        ((n, v["r2"]["point"]) for n, v in out["frames"].items() if v.get("status") == "ok"),
        key=lambda t: -t[1] if np.isfinite(t[1]) else np.inf,
    )
    out["ranking"] = [{"frame": n, "r2": r} for n, r in ranked]
    names = list(readouts)
    out["readout_similarity"] = {
        "principal_angles_mean_sq_cos": {a: {b: subspace_similarity(readouts[a], readouts[b]) for b in names} for a in names},
        "linear_cka": {a: {b: linear_cka(X @ readouts[a].T, X @ readouts[b].T) for b in names} for a in names},
        "note": "values near 1 mean the frames are read out through the same activation subspace (same geometry, different numbers)",
    }
    return out


def write_frames_json(path: Path, table: dict[str, Any]) -> None:
    path.write_text(json.dumps(table, indent=2, sort_keys=True, default=float) + "\n")
