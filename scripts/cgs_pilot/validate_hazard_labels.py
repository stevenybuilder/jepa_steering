#!/usr/bin/env python3
"""Independent hazard-label validation from RAW saved simulator records.

Motivation: a mislabeled scene (hazard not actually under the path, control
object secretly contacted, object teleported into a non-resting pose, visible
blob that is not the target, wrong episode XML) would poison every downstream
DiD. The generator already gates these at generation time, but it grades its
own homework. This script re-derives every label from the saved `.npz` cell
records in a FRESH environment built from the seed and refuses to trust any
number in `manifest.jsonl` / `summary.json` except as something to compare to.

Per seed it checks:

1. XML reproduction: `make_env(seed)` yields the same episode XML hash the
   manifest recorded (else the saved states are meaningless in this env).
2. Geometry from raw state: restore each cell's `initial_state`, read the egg
   body position and the end-effector position directly from MuJoCo, and
   recompute gripper-to-approach distance. H1 must be <= 0.03 m, H0 >= 0.12 m.
3. Resting pose: the teleported egg must be in contact with the counter, have
   ~zero velocity, and must NOT move in the three no-contact cells
   (|egg_final - egg_initial| <= 2 mm) - otherwise egg motion unrelated to the
   candidate action confounds the hazard main effect. In (H1,A1) it must move
   or register force.
4. Visibility from a fresh segmentation render: target pixel count >= 25,
   border clearance >= 16 px, and the segmentation centroid must lie within
   `--proj-tol-px` of the analytic pinhole projection of the egg's 3D centre
   through the JEPA evaluation camera (so the visible blob IS the egg where the
   physics says it is).
5. Contact identity: replay each cell and require that recorded egg-robot
   contact pairs involve a gripper finger/pad geom, and that no-contact cells
   have zero egg-robot contacts under an independent replay.
6. State contrast: H0 and H1 `initial_state` differ only within the egg's
   free-joint qpos slice.
7. Replay: replaying each cell from its saved `initial_state` with the exact
   float64 candidate actions (rebuilt via `candidate_sequence`; the saved
   `model_actions` are the float32 copy the model sees) must reproduce the saved
   `final_state` to <= 1e-8 and replicate the contact label.
   Also writes `<seed_dir>/masks/<cell_id>.npz` (egg_mask, robot_mask
   [4,256,256] bool for context+3 futures; eef_px, egg_px [4,2]; cam_pos,
   cam_mat, fovy) for token-group and coordinate-frame analyses.
8. Frames: the saved first context frame equals a fresh render of the restored
   state (<= 1 px, <= 1/255 under v0.3).

Writes `<out>/hazard_label_validation.json` plus one contact-sheet PNG per seed
(initial H0/H1 with egg bbox and the four final frames) for a human spot check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robocasa_contact_pilot as gen  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
from protocol import PROTOCOL_VERSION_V03, sha256_bytes  # noqa: E402

EGG = gen.TARGET_OBJECT


def egg_pos(env) -> np.ndarray:
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[EGG]], dtype=np.float64).copy()


def eef_pos(env) -> np.ndarray:
    obs = env._get_observations(force_update=True)
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()


def project_point(env, point_world: np.ndarray) -> tuple[float, float]:
    """Pinhole projection through the named camera, image y down, matching render_rgb's flip."""
    cam_id = env.sim.model.camera_name2id(gen.CAMERA_NAME)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    fovy = float(env.sim.model.cam_fovy[cam_id])
    rel = cam_mat.T @ (point_world - cam_pos)  # camera frame: x right, y up, z backward
    if rel[2] >= 0:
        return (float("nan"), float("nan"))
    f = 0.5 * gen.IMAGE_SIZE / np.tan(np.deg2rad(fovy) / 2.0)
    u = 0.5 * gen.IMAGE_SIZE + f * rel[0] / (-rel[2])
    v = 0.5 * gen.IMAGE_SIZE - f * rel[1] / (-rel[2])
    return (float(u), float(v))


def robot_geom_ids(env) -> list[int]:
    ids = []
    for gid in range(env.sim.model.ngeom):
        name = (env.sim.model.geom_id2name(gid) or "").lower()
        if any(tok in name for tok in ("robot", "panda", "gripper", "finger", "mobilebase")):
            ids.append(gid)
    return ids


def seg_masks(env, egg_ids: list[int], robot_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    seg = env.sim.render(height=gen.IMAGE_SIZE, width=gen.IMAGE_SIZE, camera_name=gen.CAMERA_NAME, segmentation=True)[::-1]
    return np.isin(seg[:, :, 1], egg_ids), np.isin(seg[:, :, 1], robot_ids)


def replay_with_masks(env, model_xml, start_state, actions, steps_per, egg_ids, robot_ids):
    """Generator-identical replay that also records per-model-step masks and projections."""
    gen.restore_replay_state(env, model_xml, start_state)
    cam_id = env.sim.model.camera_name2id(gen.CAMERA_NAME)
    cam_pos = np.asarray(env.sim.data.cam_xpos[cam_id], dtype=np.float64).copy()
    cam_mat = np.asarray(env.sim.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    fovy = float(env.sim.model.cam_fovy[cam_id])
    egg_m, rob_m, eef_px, egg_px, frames = [], [], [], [], [gen.render_rgb(env)]
    e, r = seg_masks(env, egg_ids, robot_ids); egg_m.append(e); rob_m.append(r)
    eef_px.append(project_point(env, eef_pos(env))); egg_px.append(project_point(env, egg_pos(env)))
    max_force, contacts, pairs = 0.0, 0, []
    for a in actions:
        sim_a = gen.model_to_sim_action(a, env.action_dim)
        for _ in range(steps_per):
            env.step(sim_a)
            m = gen.contact_metrics(env)
            max_force = max(max_force, m["max_normal_force_n"]); contacts += m["egg_robot_contact_count"]; pairs.extend(m["contact_pairs"])
        frames.append(gen.render_rgb(env))
        e, r = seg_masks(env, egg_ids, robot_ids); egg_m.append(e); rob_m.append(r)
        eef_px.append(project_point(env, eef_pos(env))); egg_px.append(project_point(env, egg_pos(env)))
    return {
        "frames": np.stack(frames), "final_state": env.sim.get_state().flatten().copy(),
        "egg_final_position": egg_pos(env), "egg_robot_contact_count": int(contacts),
        "max_normal_force_n": float(max_force), "contact_pairs": pairs,
        "egg_mask": np.stack(egg_m), "robot_mask": np.stack(rob_m),
        "eef_px": np.asarray(eef_px, dtype=np.float32), "egg_px": np.asarray(egg_px, dtype=np.float32),
        "cam_pos": cam_pos, "cam_mat": cam_mat, "fovy": fovy,
    }


def egg_qpos_slice(env) -> tuple[int, int]:
    joint = env.objects[EGG].joints[0]
    jid = env.sim.model.joint_name2id(joint)
    start = int(env.sim.model.jnt_qposadr[jid])
    return start, start + 7


def egg_qvel_norm(env) -> float:
    joint = env.objects[EGG].joints[0]
    return float(np.linalg.norm(env.sim.data.get_joint_qvel(joint)))


def validate_seed(seed_dir: Path, seed: int, proj_tol_px: float, sheet_dir: Path) -> dict:
    rows = [json.loads(l) for l in (seed_dir / "manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if int(r["seed"]) == seed]
    cells = {(int(r["hazard"]), int(r["candidate_action"])): r for r in rows}
    checks: dict[str, object] = {"seed": seed, "n_cells": len(rows)}
    fails: list[str] = []
    env = gen.make_env(seed)
    try:
        env.reset()
        model_xml = env.sim.model.get_xml()
        xml_hash = sha256_bytes(model_xml.encode("utf-8"))
        manifest_xml = {r["model_xml_sha256"] for r in rows}
        checks["xml_reproduced"] = manifest_xml == {xml_hash}
        if not checks["xml_reproduced"]:
            fails.append("xml_hash_mismatch")
            return {**checks, "passed": False, "failures": fails}

        qs, qe = egg_qpos_slice(env)
        egg_ids = [env.sim.model.geom_name2id(n) for n in env.objects[EGG].visual_geoms]
        rob_ids = robot_geom_ids(env)
        mask_dir = seed_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        replay_diag = {}
        per_cell = {}
        sheet_tiles = {}
        egg_initial = {}
        for (h, a), row in sorted(cells.items()):
            cell = np.load(seed_dir / row["artifact"])
            init_state = np.asarray(cell["initial_state"], dtype=np.float64)
            saved_actions = np.asarray(cell["model_actions"], dtype=np.float32)
            # The generator simulates float64 candidate actions and saves them as
            # float32; replaying the float32 copy drifts (diagnose_replay_divergence.py:
            # -0.03 is not float32-representable). Rebuild the exact float64 sequence
            # and assert it rounds to the saved array.
            actions = gen.candidate_sequence(a, int(row["horizon"]))
            if not np.array_equal(actions.astype(np.float32), saved_actions):
                fails.append(f"h{h}a{a}:saved_actions_do_not_match_candidate_sequence")
                actions = saved_actions.astype(np.float64)
            steps_per = int(row.get("sim_steps_per_model_step", 5))

            gen.restore_replay_state(env, model_xml, init_state)
            p_egg = egg_pos(env)
            p_eef = eef_pos(env)
            egg_initial[h] = p_egg
            approach_dist = float(np.linalg.norm((p_egg + np.array([0.0, 0.0, 0.075])) - p_eef))
            on_counter = bool(OU.check_obj_fixture_contact(env, EGG, env.counter))
            vel = egg_qvel_norm(env)
            vis = gen.target_visibility(env)
            proj = project_point(env, p_egg)
            centroid = vis["target_centroid_xy"]
            proj_err = float(np.hypot(centroid[0] - proj[0], centroid[1] - proj[1])) if centroid else float("inf")
            fresh_frame = gen.render_rgb(env)
            saved_ctx = np.asarray(cell["context_frames"])[-1]
            fd = np.abs(fresh_frame.astype(np.int16) - saved_ctx.astype(np.int16))
            frame_diff_px = int((fd.max(axis=-1) > 0).sum())
            frame_diff_max = int(fd.max())

            replay = replay_with_masks(env, model_xml, init_state, actions, steps_per, egg_ids, rob_ids)
            np.savez_compressed(
                mask_dir / f"{row['cell_id']}.npz",
                egg_mask=replay["egg_mask"], robot_mask=replay["robot_mask"],
                eef_px=replay["eef_px"], egg_px=replay["egg_px"],
                cam_pos=replay["cam_pos"], cam_mat=replay["cam_mat"], fovy=np.float64(replay["fovy"]),
                frame_index_meaning=np.array(["context", "future1", "future2", "future3"][: len(replay["egg_mask"])]),
            )
            final_err = float(np.max(np.abs(replay["final_state"] - np.asarray(cell["final_state"]))))
            # With float64 actions replay is bit-exact across processes; any
            # residual error is reported and is a failure above 1e-8.
            replay_diag[f"h{h}a{a}"] = {"final_state_err": final_err, "force_saved": float(row["max_normal_force_n"]), "force_replay": replay["max_normal_force_n"]}
            egg_moved = float(np.linalg.norm(replay["egg_final_position"] - p_egg))
            saved_egg_moved = float(np.linalg.norm(np.asarray(cell["egg_final_position"]) - np.asarray(cell["egg_initial_position"])))
            pairs = replay["contact_pairs"]
            finger_like = all(any(t in f"{p[0]} {p[1]}".lower() for t in ("finger", "pad", "gripper")) for p in pairs)
            contact = replay["egg_robot_contact_count"] > 0

            expected_contact = (h, a) == (1, 1)
            c = {
                "approach_dist_m": approach_dist,
                "manifest_approach_dist_m": float(row["gripper_to_target_approach_m"]),
                "on_counter": on_counter,
                "egg_qvel_norm": vel,
                "visible_px": vis["target_visible_pixels"],
                "border_px": vis["target_border_clearance_px"],
                "seg_centroid_xy": centroid,
                "projected_centroid_xy": list(proj),
                "projection_error_px": proj_err,
                "fresh_vs_saved_context_frame": {"differing_pixels": frame_diff_px, "max_abs": frame_diff_max},
                "replay_final_state_err": final_err,
                "egg_displacement_m_replay": egg_moved,
                "egg_displacement_m_saved": saved_egg_moved,
                "contact": contact,
                "max_force_n": replay["max_normal_force_n"],
                "contact_pairs": pairs[:6],
                "contact_pairs_are_gripper": finger_like,
                "expected_contact": expected_contact,
            }
            tag = f"h{h}a{a}"
            if h == 1 and approach_dist > 0.03:
                fails.append(f"{tag}:hazard_not_under_path({approach_dist:.3f}m)")
            if h in (0, 2) and approach_dist < 0.12:
                fails.append(f"{tag}:control_too_close({approach_dist:.3f}m)")
            if abs(approach_dist - c["manifest_approach_dist_m"]) > 1e-6:
                fails.append(f"{tag}:manifest_distance_disagrees")
            if not on_counter:
                fails.append(f"{tag}:egg_not_resting_on_counter")
            if vel > 1e-6:
                fails.append(f"{tag}:egg_moving_at_t0")
            if vis["target_visible_pixels"] < 25 or vis["target_border_clearance_px"] < 16:
                fails.append(f"{tag}:visibility")
            if not np.isfinite(proj_err) or proj_err > proj_tol_px:
                fails.append(f"{tag}:seg_blob_not_at_egg_projection({proj_err:.1f}px)")
            if frame_diff_px > 1 or frame_diff_max > 1:
                fails.append(f"{tag}:saved_context_frame_mismatch")
            saved_contact = int(row["egg_robot_contact_count"]) > 0
            if contact != saved_contact:
                fails.append(f"{tag}:contact_label_not_replicated_under_replay")
            if final_err > 1e-8:
                fails.append(f"{tag}:final_state_not_reproduced({final_err:.2e})")
            if contact != expected_contact:
                fails.append(f"{tag}:contact_pattern({contact} vs {expected_contact})")
            if not expected_contact and (egg_moved > 0.002 or saved_egg_moved > 0.002):
                fails.append(f"{tag}:egg_moved_without_contact({max(egg_moved, saved_egg_moved):.4f}m)")
            if expected_contact and not (egg_moved > 0.002 or replay["max_normal_force_n"] >= 5.0):
                fails.append(f"{tag}:contact_cell_without_physical_effect")
            if pairs and not finger_like:
                fails.append(f"{tag}:contact_not_with_gripper")
            per_cell[tag] = c
            sheet_tiles[tag] = (saved_ctx, vis["target_bbox_xyxy"], replay["frames"][-1])

        # State contrast: H0 vs H1 initial states differ only in egg qpos slice.
        s0 = np.asarray(np.load(seed_dir / cells[(0, 0)]["artifact"])["initial_state"])
        s1 = np.asarray(np.load(seed_dir / cells[(1, 0)]["artifact"])["initial_state"])
        changed = np.flatnonzero(np.abs(s1 - s0) > 1e-12)
        # flattened state = [time, qpos, qvel, act...]; qpos starts at index 1
        egg_flat = set(range(1 + qs, 1 + qe))
        contrast_ok = len(changed) >= 1 and set(changed.tolist()).issubset(egg_flat)
        checks["state_contrast"] = {"changed_indices": changed.tolist(), "egg_qpos_flat_indices": sorted(egg_flat), "ok": contrast_ok}
        if not contrast_ok:
            fails.append("state_contrast_not_egg_only")
        if (2, 0) in cells:  # H0' null-factor cells must also differ from H1 only in the egg joint
            s2 = np.asarray(np.load(seed_dir / cells[(2, 0)]["artifact"])["initial_state"])
            changed2 = np.flatnonzero(np.abs(s2 - s1) > 1e-12)
            ok2 = len(changed2) >= 1 and set(changed2.tolist()).issubset(egg_flat)
            checks["null_factor_state_contrast"] = {"changed_indices": changed2.tolist(), "ok": ok2}
            if not ok2:
                fails.append("null_factor_state_contrast_not_egg_only")
            sep2 = float(np.hypot(*(np.asarray(per_cell["h2a0"]["seg_centroid_xy"]) - np.asarray(per_cell["h0a0"]["seg_centroid_xy"]))))
            sep2h = float(np.hypot(*(np.asarray(per_cell["h2a0"]["seg_centroid_xy"]) - np.asarray(per_cell["h1a0"]["seg_centroid_xy"]))))
            checks["null_factor_centroid_separation_px"] = {"to_h0": sep2, "to_h1": sep2h}
            if min(sep2, sep2h) < 16:
                fails.append(f"null_factor_centroid_separation({min(sep2, sep2h):.1f}px)")
        sep_px = float(np.hypot(*(np.asarray(per_cell["h1a0"]["seg_centroid_xy"]) - np.asarray(per_cell["h0a0"]["seg_centroid_xy"]))))
        checks["hazard_control_centroid_separation_px"] = sep_px
        if sep_px < 16:
            fails.append(f"centroid_separation({sep_px:.1f}px)")
        checks["cells"] = per_cell
        checks["replay_diagnostics"] = replay_diag
        checks["cross_process_replay_exact"] = bool(all(v["final_state_err"] <= 1e-8 for v in replay_diag.values()))

        # Contact sheet.
        sheet_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for tag in ("h0a0", "h1a0"):
            img, bbox, _ = sheet_tiles[tag]
            im = Image.fromarray(img).convert("RGB")
            if bbox:
                ImageDraw.Draw(im).rectangle(bbox, outline=(255, 0, 0), width=1)
            tiles.append(im)
        for tag in ("h0a0", "h0a1", "h1a0", "h1a1"):
            tiles.append(Image.fromarray(sheet_tiles[tag][2]).convert("RGB"))
        W = gen.IMAGE_SIZE
        sheet = Image.new("RGB", (W * 6, W + 14), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        labels = ["init H0 (ctrl)", "init H1 (hazard)", "final h0a0", "final h0a1", "final h1a0", "final h1a1"]
        for i, (im, lab) in enumerate(zip(tiles, labels)):
            sheet.paste(im, (i * W, 14))
            d.text((i * W + 2, 1), f"s{seed} {lab}", fill=(0, 0, 0))
        sheet.save(sheet_dir / f"seed_{seed}_contact_sheet.png")
    finally:
        env.close()
    checks["failures"] = fails
    checks["passed"] = not fails
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-group", default="egg", help="RoboCasa obj_groups of the family (must match generation)")
    ap.add_argument("--control-min-m", type=float, default=0.12, help="control offset used at generation")
    ap.add_argument("--root", type=Path, required=True, help="dir containing seed_<s>/ subdirs, or a single stimulus dir")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--proj-tol-px", type=float, default=6.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sheet-dir", type=Path, default=None)
    args = ap.parse_args()
    gen.OBJ_GROUP = args.obj_group
    gen.CONTROL_MIN_M = args.control_min_m
    sheet_dir = args.sheet_dir or args.output.parent / "contact_sheets"

    seed_dirs: list[tuple[int, Path]] = []
    if (args.root / "manifest.jsonl").exists():
        seeds = sorted({int(json.loads(l)["seed"]) for l in (args.root / "manifest.jsonl").read_text().splitlines()})
        seed_dirs = [(s, args.root) for s in seeds]
    else:
        for d in sorted(args.root.glob("seed_*")):
            if (d / "manifest.jsonl").exists():
                seed_dirs.append((int(d.name.split("_")[1]), d))
    if args.seeds:
        seed_dirs = [(s, d) for s, d in seed_dirs if s in set(args.seeds)]

    results = []
    existing = {}
    if args.output.exists():
        existing = {r["seed"]: r for r in json.loads(args.output.read_text()).get("seeds", [])}
    for seed, d in seed_dirs:
        if seed in existing:
            results.append(existing[seed])
            continue
        try:
            r = validate_seed(d, seed, args.proj_tol_px, sheet_dir)
        except Exception as exc:  # keep going; record the crash as a failure
            r = {"seed": seed, "passed": False, "failures": [f"exception:{exc!r}"]}
        print(json.dumps({"seed": seed, "passed": r["passed"], "failures": r.get("failures")}), flush=True)
        results.append(r)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"protocol": PROTOCOL_VERSION_V03, "proj_tol_px": args.proj_tol_px, "seeds": results}, indent=2) + "\n")
    n_pass = sum(1 for r in results if r["passed"])
    print(f"VALIDATED {n_pass}/{len(results)} seeds pass independent hazard-label checks")


if __name__ == "__main__":
    main()
