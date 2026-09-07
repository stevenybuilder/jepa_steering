#!/usr/bin/env python
"""Independent re-derivation of the MetaDrive factorial labels from the saved cell records (fresh process).

For every seed (and arm) it rebuilds the scene from the seed, replays the recorded raw per-sim-step actions from the
same prefix, and refuses to trust anything in manifest.jsonl except as a value to compare against:

1. Ego-state replay: recorded ``ego_states`` vs the fresh replay. Reported as max |diff| and a bit-exact flag. The
   generator runs one seed per process by default and the validator mirrors the generator's in-process episode order
   (hazard-free baselines A0, A1, then the 8 cells of arm A, then arm B), because Bullet's per-episode result depends on
   the in-process history (measured 2026-09-02: 1e-12 hazard-free, up to 1e-8 post-contact between two runs of the same
   cell in one process; bit-exact across fresh processes with the same history). Gate: bit-exact (``--state-tol`` relaxes it).
2. Frames: differing pixel count and max |diff| per frame vs the saved frames. Preregistered tolerance: <= 16 differing
   pixels per frame with |diff| <= 8/255 (renderer jitter class measured in the spike); anything worse is flagged.
3. Contact flags and minimum gap re-derived from the replay and compared with the manifest; the expected pattern per
   arm (contact only in the solid in-lane cell under throttle) is checked.
4. Visibility from a fresh semantic render: in-lane cells (levels 1, 3) silhouette covers >= 6 DINOv3 16x16 patches in
   the context frame (silhouette and footprint bboxes both reported); all four hazard poses fully in frame; the bbox
   centre must lie within ``--proj-tol-px`` (6) of the analytic projection of
   the hazard's 3D centre through the camera (the mask centroid offset is reported too; for the cone it is biased
   downwards by the triangular silhouette).
5. Ghost and off-path cells: zero contact and ego states within 1e-9 of the hazard-free replay of the same actions.
6. H0' null: |c(H0') - c(H1)| within 10 % of |c(H0) - c(H1)| (mask centroids), plus >= 16 px from both.
7. Solid vs ghost arm pairing (when both arms are present): frames identical (<= jitter) up to the first contact.

Writes ``<out>`` JSON of the same shape as validate_hazard_labels.py ({"protocol", "seeds": [{seed, passed, failures,
cells...}]}) so merge_heldout.py can consume it (one file per arm), and a contact sheet PNG per seed and arm.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metadrive_hazard_pilot as gen  # noqa: E402

# Tolerances amended by the coordinator 2026-09-02 (preregistered before the full run, after the 20-seed pilot showed
# EGL jitter of 9-14/255 on <= 5 px and no-contact deviations up to 3e-7 m): |diff| <= 16/255 on <= 16 px/frame; 1e-6 m.
PIX_TOL_ABS = 16
PIX_TOL_COUNT = 16
GHOST_STATE_TOL = 1e-6
NULL_MATCH_FRAC = 0.10
MIN_SEP_PX = 16.0


def load_rows(seed_dir: Path, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in (seed_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if int(r["seed"]) == seed]


def frame_diff(a: np.ndarray, b: np.ndarray) -> tuple[list[int], int]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return (d.max(-1) > 0).reshape(d.shape[0], -1).sum(1).tolist(), int(d.max())


def validate_seed(env, root: Path, seed: int, arms: list[str], proj_tol: float, state_tol: float, sheet_dir: Path | None,
                  mirror_history: bool = True, free_body_tol: float | None = None) -> dict[str, dict[str, Any]]:
    """Replays baselines + every cell of every arm in the generator's order. Returns {arm: record}.

    free_body_tol (v0.9): drift tolerance (m, 3-D) for a ``hazard_body == dynamic`` hazard in cells WITHOUT contact,
    used instead of the 1e-6 m ghost/static rule; None -> 1e-6 (unchanged). For a dynamic body, "physical effect" of a
    contact is ego slowdown OR ego-state deviation from the hazard-free replay (> --ghost-tol) OR body displacement."""
    arm_dirs = {a: root / f"arm{a}" / f"seed_{seed}" for a in arms}
    rows = {a: load_rows(arm_dirs[a], seed) for a in arms}
    for a in arms:
        if len(rows[a]) != 8:
            return {a: {"seed": seed, "arm": a, "passed": False, "failures": [f"incomplete_octet({len(rows[a])})"]} for a in arms}
    row0 = rows[arms[0]][0]
    map_cfg = str(row0["map_cfg"])
    replayed_twice = False  # the generator runs each cell once; its replay_* fields come from a fresh-process pass with this same sequence
    # stimulus factors re-derived from the saved rows only: hazard_body, fov_deg, settle_steps, prefix_throttle, in-lane lateral
    # (v0.8 rows carry none of the v0.9 keys -> v0.8 values); the generator's module globals are set accordingly
    dist, prefix_throttle, lat_inlane = gen.apply_row_factors(row0)
    dynamic = gen.HAZARD_BODY == "dynamic"
    drift_tol = float(free_body_tol) if (dynamic and free_body_tol is not None) else 1e-6
    chunks = {a: np.repeat(np.asarray([gen.ACTION_CHUNKS[a]], dtype=np.float64), gen.HORIZON, axis=0) for a in (0, 1)}
    base = {a: gen.run_episode(env, seed, map_cfg, None, chunks[a], prefix_throttle=prefix_throttle) for a in (0, 1)}
    prefix_travel = gen.prefix_travel_of(base[0]["states"])
    out: dict[str, dict[str, Any]] = {}
    contact_first_frames: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for arm in arms:
        adir = arm_dirs[arm]
        fails: list[str] = []
        per_cell: dict[str, Any] = {}
        cells = {(int(r["hazard"]), int(r["candidate_action"])): r for r in rows[arm]}
        vis: dict[int, dict[str, Any]] = {}
        ctx_sha: dict[tuple[int, int], str] = {}
        tiles: dict[str, np.ndarray] = {}
        for level in (0, 1, 2, 3):
            kind, where = gen.HAZARD_LEVELS[level]
            solid = gen.ARMS[arm][kind]
            lat = gen.level_lateral(env, where, lat_inlane)
            hz = {"kind": kind, "s_ahead": dist, "lateral": lat, "solid": solid, "prefix_travel": prefix_travel}
            for a in (0, 1):
                row = cells[(level, a)]
                tag = f"h{level}a{a}"
                cell = np.load(adir / row["artifact"])
                raw_saved = np.asarray(cell["raw_actions"], dtype=np.float64)
                raw_expected = gen.raw_actions_from_chunks(chunks[a])
                if not np.array_equal(raw_saved.astype(np.float32), raw_expected.astype(np.float32)):
                    fails.append(f"{tag}:saved_actions_do_not_match_chunk")
                ep = gen.run_episode(env, seed, map_cfg, hz, chunks[a], prefix_throttle=prefix_throttle, baseline_frames=base[a]["frames"])
                if replayed_twice and mirror_history:
                    gen.run_episode(env, seed, map_cfg, hz, chunks[a], prefix_throttle=prefix_throttle, with_masks=False)
                saved_states = np.asarray(cell["ego_states"], dtype=np.float64)
                saved_frames = np.concatenate([cell["context_frames"], cell["true_future_frames"]])
                st_err = float(np.abs(ep["states"] - saved_states).max())
                bit_exact = bool(np.array_equal(ep["states"], saved_states))
                px_counts, px_max = frame_diff(ep["frames"], saved_frames)
                ctx_shared = str(row.get("context_frame_source", "")) != ""
                if ctx_shared:  # context frame is the level's shared render; compare the replay's own context render separately
                    px_counts_ctx, px_max_ctx = frame_diff(ep["frames"][:1], saved_frames[:1])
                cs = gen.contact_summary(ep)
                v = gen.bbox_stats(ep["hazard_mask"][0])
                fp = gen.bbox_stats(ep["footprint_mask"][0])
                vis[level] = v
                proj = ep["hazard_px"][0]
                bbox = v["target_bbox_xyxy"]
                bbox_centre = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] if bbox else None
                proj_err_bbox = float(np.hypot(bbox_centre[0] - proj[0], bbox_centre[1] - proj[1])) if bbox_centre and proj[2] else float("inf")
                proj_err_centroid = (float(np.hypot(v["target_centroid_xy"][0] - proj[0], v["target_centroid_xy"][1] - proj[1]))
                                     if v["target_centroid_xy"] and proj[2] else float("inf"))
                expected_contact = bool(a == 1 and solid and where == "in_lane")
                contact = cs["egg_robot_contact_count"] > 0
                base_dev = float(np.abs(ep["states"] - base[a]["states"]).max())
                base_dev_saved = float(np.abs(saved_states - base[a]["states"]).max())
                ctx_sha[(level, a)] = gen.sha256_array(saved_frames[0])
                hz_moved = float(np.hypot(*(np.asarray(ep["hazard_final_pose"][:2]) - np.asarray(ep["hazard_pose"][:2]))))
                hz_moved_xyz = float(np.linalg.norm(np.asarray(ep["hazard_final_pose"][:3]) - np.asarray(ep["hazard_pose"][:3])))
                moved = hz_moved_xyz if dynamic else hz_moved  # a knocked-over body also drops in z
                c = {
                    "replay_state_max_err": st_err, "replay_state_bit_exact": bit_exact,
                    "replay_frame_differing_px": px_counts, "replay_frame_max_abs": px_max,
                    "context_frame_shared": ctx_shared, "manifest_context_render_jitter": [row.get("context_render_jitter_px"), row.get("context_render_jitter_max_abs")],
                    "contact": contact, "expected_contact": expected_contact, "manifest_contact": int(row["egg_robot_contact_count"]) > 0,
                    "first_contact_step": cs["first_contact_step"], "manifest_first_contact_step": row.get("first_contact_step"),
                    "min_gap_m": cs["min_gap_m"], "manifest_min_gap_m": float(row["min_gap_m"]),
                    "min_center_distance_m": cs["min_center_distance_m"], "context_gap_m": cs["context_longitudinal_gap_m"],
                    "context_speed_mps": cs["context_speed_mps"], "final_speed_mps": cs["final_speed_mps"],
                    "crash_prefix_any": cs["crash_prefix_any"],
                    "visible_px": v["target_visible_pixels"], "bbox_xyxy": bbox, "bbox_wh": [v["bbox_w"], v["bbox_h"]],
                    "silhouette_patches": v["patches"], "manifest_silhouette_patches": row.get("silhouette_patches"),
                    "footprint_bbox_xyxy": fp["target_bbox_xyxy"], "footprint_wh": [fp["bbox_w"], fp["bbox_h"]], "footprint_px": fp["target_visible_pixels"],
                    "manifest_footprint_wh": row.get("footprint_wh"), "manifest_bbox_wh": row.get("bbox_wh"),
                    "border_px": v["target_border_clearance_px"], "seg_centroid_xy": v["target_centroid_xy"], "bbox_centre_xy": bbox_centre,
                    "projected_centre_xy": [float(proj[0]), float(proj[1])], "projection_error_bbox_px": proj_err_bbox,
                    "projection_error_centroid_px": proj_err_centroid,
                    "baseline_state_max_dev": base_dev, "baseline_state_max_dev_saved": base_dev_saved,
                    "hazard_z_range": [cs["hazard_z_min"], cs["hazard_z_max"]], "hazard_moved_m": hz_moved, "hazard_moved_xyz_m": hz_moved_xyz,
                    "hazard_settle_drift_m": ep.get("hazard_settle_drift_m"), "manifest_hazard_settle_drift_m": row.get("hazard_settle_drift_m"),
                    "free_body_drift_tol_m": drift_tol if dynamic else None, "solid": solid, "kind": kind,
                }
                if st_err > state_tol or (state_tol == 0.0 and not bit_exact):
                    fails.append(f"{tag}:ego_state_not_reproduced({st_err:.2e})")
                if px_max > PIX_TOL_ABS or max(px_counts) > PIX_TOL_COUNT:
                    fails.append(f"{tag}:frame_jitter_exceeds_tolerance(max={px_max},npx={max(px_counts)})")
                if contact != c["manifest_contact"]:
                    fails.append(f"{tag}:contact_label_not_replicated")
                if contact != expected_contact:
                    fails.append(f"{tag}:contact_pattern({contact} vs {expected_contact})")
                if abs(cs["min_gap_m"] - c["manifest_min_gap_m"]) > 1e-6:
                    fails.append(f"{tag}:manifest_min_gap_disagrees")
                if cs["crash_prefix_any"]:
                    fails.append(f"{tag}:contact_during_prefix")
                if level in (1, 3) and v["patches"] < gen.MIN_PATCHES:
                    fails.append(f"{tag}:silhouette_below_{gen.MIN_PATCHES}_patches({v['patches']};bbox {v['bbox_w']}x{v['bbox_h']})")
                if row.get("bbox_wh") is not None and list(row["bbox_wh"]) != [v["bbox_w"], v["bbox_h"]]:
                    fails.append(f"{tag}:manifest_bbox_disagrees")
                if row.get("hazard_bbox_xyxy") is not None and list(row["hazard_bbox_xyxy"]) != list(bbox):
                    fails.append(f"{tag}:manifest_hazard_bbox_disagrees")
                if "min_distance_m" in row and abs(float(row["min_distance_m"]) - cs["min_gap_m"]) > 1e-6:
                    fails.append(f"{tag}:manifest_min_distance_disagrees")
                if "contact_step" in row and row["contact_step"] != cs["first_contact_step"]:
                    fails.append(f"{tag}:manifest_contact_step_disagrees")
                if float(row.get("replay_max_state_error", 0.0)) != 0.0:
                    fails.append(f"{tag}:generator_replay_not_bit_exact({row['replay_max_state_error']:.2e})")
                if v["target_visible_pixels"] == 0 or v["target_border_clearance_px"] < 1:
                    fails.append(f"{tag}:hazard_not_fully_in_frame")
                if not np.isfinite(proj_err_bbox) or proj_err_bbox > proj_tol:
                    fails.append(f"{tag}:mask_not_at_projected_pose({proj_err_bbox:.1f}px)")
                if not expected_contact:
                    if base_dev > GHOST_STATE_TOL:
                        fails.append(f"{tag}:nocontact_cell_deviates_from_hazard_free({base_dev:.2e})")
                    if moved > drift_tol:
                        fails.append(f"{tag}:hazard_moved_without_contact({moved:.2e}m>{drift_tol:.0e})")
                elif cs["first_contact_step"] is None or cs["first_contact_step"] > gen.HORIZON * gen.SIM_STEPS_PER_MODEL_STEP - 1:
                    fails.append(f"{tag}:contact_not_within_horizon")
                else:
                    slowed = cs["final_speed_mps"] < cs["context_speed_mps"]
                    # v0.9: a free body absorbs little momentum (ego 8.3 -> 8.9 m/s under throttle, F18); the physical effect of the
                    # contact is then the displaced body or the ego trajectory leaving the hazard-free replay
                    effect = slowed or (dynamic and (moved > drift_tol or base_dev > GHOST_STATE_TOL))
                    c["physical_effect"] = {"ego_slowed": bool(slowed), "body_displaced": bool(moved > drift_tol), "ego_deviates_from_hazard_free": bool(base_dev > GHOST_STATE_TOL)}
                    if not effect:
                        fails.append(f"{tag}:contact_without_physical_effect")
                per_cell[tag] = c
                tiles[tag] = np.concatenate([saved_frames[0], ep["frames"][-1]], axis=1)
                if expected_contact:
                    contact_first_frames.setdefault(arm, {})[(level, a)] = ep["frames"]
                else:
                    contact_first_frames.setdefault(arm, {})[(level, a)] = ep["frames"]
        # scene-level checks
        if not all(ctx_sha[(l, 0)] == ctx_sha[(l, 1)] for l in (0, 1, 2, 3)):
            fails.append("context_frame_differs_across_actions")
        c0, c1, c2 = (np.asarray(vis[l]["target_centroid_xy"] or [np.nan, np.nan], dtype=np.float64) for l in (0, 1, 2))
        d_h0 = float(np.linalg.norm(c0 - c1)); d_null = float(np.linalg.norm(c2 - c1)); d_null_h0 = float(np.linalg.norm(c2 - c0))
        null = {"hazard_to_control_px": d_h0, "null_to_hazard_px": d_null, "null_to_control_px": d_null_h0,
                "match_frac": abs(d_null - d_h0) / max(d_h0, 1e-9) if np.isfinite(d_h0) else float("inf")}
        if not (np.isfinite(null["match_frac"]) and null["match_frac"] <= NULL_MATCH_FRAC):
            fails.append(f"null_displacement_mismatch({null['match_frac']:.3f})")
        if min(d_h0, d_null, d_null_h0) < MIN_SEP_PX:
            fails.append(f"centroid_separation({min(d_h0, d_null, d_null_h0):.1f}px)")
        rec = {"seed": seed, "arm": arm, "map_cfg": map_cfg, "hazard_dist_m": dist, "prefix_travel_m": prefix_travel,
               "protocol_version": row0.get("protocol_version"), "hazard_body": gen.HAZARD_BODY, "prefix_throttle": prefix_throttle,
               "hazard_lateral_offset_m": lat_inlane, "fov_deg": gen.FOV_DEG, "settle_steps": gen.SETTLE_STEPS,
               "randomized_factors": row0.get("randomized_factors"), "free_body_drift_tol_m": drift_tol if dynamic else None,
               "replayed_twice_mirrored": bool(replayed_twice and mirror_history), "cells": per_cell, "null_factor": null,
               "hazard_control_centroid_separation_px": d_h0,
               "cross_process_replay_exact": all(c["replay_state_bit_exact"] for c in per_cell.values()),
               "max_frame_jitter_px": max(max(c["replay_frame_differing_px"]) for c in per_cell.values()),
               "max_frame_jitter_abs": max(c["replay_frame_max_abs"] for c in per_cell.values())}
        if sheet_dir is not None:
            write_sheet(sheet_dir, seed, arm, tiles, vis)
        rec["failures"] = fails
        rec["passed"] = not fails
        out[arm] = rec
    # arm pairing: identical frames until first contact for the same (level, action)
    if len(arms) == 2 and all(a in contact_first_frames for a in arms):
        pair = {}
        for key in contact_first_frames[arms[0]]:
            fa, fb = contact_first_frames[arms[0]][key], contact_first_frames[arms[1]][key]
            ca = out[arms[0]]["cells"][f"h{key[0]}a{key[1]}"]; cb = out[arms[1]]["cells"][f"h{key[0]}a{key[1]}"]
            firsts = [x for x in (ca["first_contact_step"], cb["first_contact_step"]) if x is not None]
            upto = 1 + (min(firsts) // gen.SIM_STEPS_PER_MODEL_STEP) if firsts else fa.shape[0]  # frames strictly before contact
            counts, mx = frame_diff(fa[:upto], fb[:upto])
            pair[f"h{key[0]}a{key[1]}"] = {"frames_compared": int(upto), "differing_px": counts, "max_abs": mx}
            if mx > PIX_TOL_ABS or max(counts) > PIX_TOL_COUNT:
                for a in arms:
                    out[a]["failures"].append(f"h{key[0]}a{key[1]}:arm_frames_differ_before_contact(max={mx},npx={max(counts)})")
                    out[a]["passed"] = False
        for a in arms:
            out[a]["arm_pairing"] = pair
    return out


def write_sheet(sheet_dir: Path, seed: int, arm: str, tiles: dict[str, np.ndarray], vis: dict[int, dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover
        return
    sheet_dir.mkdir(parents=True, exist_ok=True)
    W = gen.IMAGE_SIZE
    order = [f"h{l}a{a}" for l in (0, 1, 2, 3) for a in (0, 1)]
    sheet = Image.new("RGB", (2 * W * 4, 2 * (W + 14)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, tag in enumerate(order):
        r, c = divmod(i, 4)
        im = Image.fromarray(tiles[tag])
        bbox = vis[int(tag[1])]["target_bbox_xyxy"]
        if bbox:
            ImageDraw.Draw(im).rectangle(bbox, outline=(255, 0, 0), width=1)
        sheet.paste(im, (c * 2 * W, r * (W + 14) + 14))
        d.text((c * 2 * W + 2, r * (W + 14) + 1), f"s{seed} arm{arm} {tag} ctx | final", fill=(0, 0, 0))
    sheet.save(sheet_dir / f"seed_{seed}_arm{arm}_contact_sheet.png")


def discover_seeds(root: Path, arms: list[str]) -> list[int]:
    seeds = None
    for a in arms:
        s = {int(d.name.split("_")[1]) for d in (root / f"arm{a}").glob("seed_*") if (d / "manifest.jsonl").exists()}
        seeds = s if seeds is None else seeds & s
    return sorted(seeds or [])


def main() -> None:
    global PIX_TOL_ABS, PIX_TOL_COUNT, GHOST_STATE_TOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="factorial dir containing armA/ armB/ (each with seed_<s>/)")
    ap.add_argument("--arms", default="AB")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--output", type=Path, default=None, help="output JSON prefix (default <root>/_validation/hazard_label_validation); writes <output>.json (combined, {seeds:[{seed,passed,failures,arms}]}) and <output>_arm<X>.json per arm")
    ap.add_argument("--done-log", type=Path, default=Path("/root/cgs-pilot/logs/metadrive_validate_labels.log"))
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--proj-tol-px", type=float, default=6.0)
    ap.add_argument("--state-tol", type=float, default=0.0, help="0 = require bit-exact ego states")
    ap.add_argument("--pix-tol-abs", type=int, default=PIX_TOL_ABS)
    ap.add_argument("--pix-tol-count", type=int, default=PIX_TOL_COUNT)
    ap.add_argument("--ghost-tol", type=float, default=GHOST_STATE_TOL)
    ap.add_argument("--free-body-tol", type=float, default=None, help="v0.9: 3-D drift tolerance (m) for a dynamic (free) hazard body in no-contact cells, instead of the 1e-6 m ghost rule; also the 'body displaced' threshold of the contact physical-effect check")
    ap.add_argument("--procs", type=int, default=1, help="fresh subprocess per seed, this many in parallel")
    ap.add_argument("--worker", action="store_true", help="internal: validate the given seeds in this process")
    ap.add_argument("--no-mirror-history", action="store_true")
    args = ap.parse_args()
    PIX_TOL_ABS, PIX_TOL_COUNT, GHOST_STATE_TOL = args.pix_tol_abs, args.pix_tol_count, args.ghost_tol
    arms = list(args.arms)
    if args.output is None:
        args.output = args.root / "_validation" / "hazard_label_validation"
    seeds = args.seeds or discover_seeds(args.root, arms)
    sheet_dir = args.sheet_dir or args.output.parent / "contact_sheets"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.worker:
        env = gen.make_env(num_scenarios=gen.scenario_count(seeds))
        results = []
        for seed in seeds:
            t0 = time.time()
            try:
                r = validate_seed(env, args.root, seed, arms, args.proj_tol_px, args.state_tol, sheet_dir, not args.no_mirror_history, args.free_body_tol)
            except Exception as exc:  # keep going; record the crash
                r = {a: {"seed": seed, "arm": a, "passed": False, "failures": [f"exception:{exc!r}"]} for a in arms}
            for a in arms:
                r[a]["elapsed_s"] = time.time() - t0
            results.append({"seed": seed, "arms": r})
            print(json.dumps({"seed": seed, **{a: {"passed": r[a]["passed"], "failures": r[a]["failures"]} for a in arms}}), flush=True)
        env.close()
        Path(str(args.output) + f"_worker_{seeds[0]}_{seeds[-1]}.json").write_text(json.dumps(results, indent=1) + "\n")
        return

    # driver: one fresh process per seed (matches the generator's one-seed-per-process default), --procs in parallel
    cmds = [[sys.executable, __file__, "--root", str(args.root), "--arms", args.arms, "--seeds", str(s), "--output", str(args.output),
             "--sheet-dir", str(sheet_dir), "--proj-tol-px", str(args.proj_tol_px), "--state-tol", str(args.state_tol), "--pix-tol-abs", str(args.pix_tol_abs),
             "--pix-tol-count", str(args.pix_tol_count), "--ghost-tol", str(args.ghost_tol), "--worker"]
            + (["--no-mirror-history"] if args.no_mirror_history else [])
            + (["--free-body-tol", str(args.free_body_tol)] if args.free_body_tol is not None else []) for s in seeds]
    log_dir = args.output.parent / "logs"
    t0 = time.time()
    gen.spawn_workers(cmds, args.procs, log_dir)
    wall = time.time() - t0
    per_seed: dict[int, dict[str, Any]] = {}
    for p in sorted(args.output.parent.glob(args.output.name + "_worker_*.json")):
        for rec in json.loads(p.read_text()):
            per_seed[int(rec["seed"])] = rec["arms"]
        p.unlink()
    combined = []
    # protocol string of the DATA (rows): v0.9 when the generator ran with any v0.9 flag, else the generator default
    row_protocols = {r.get("protocol_version") for s in per_seed.values() for r in s.values() if r.get("protocol_version")}
    protocol = row_protocols.pop() if len(row_protocols) == 1 else gen.PROTOCOL_VERSION_DRIVE
    for a in arms:
        recs = [per_seed[s][a] for s in sorted(per_seed) if a in per_seed[s]]
        Path(str(args.output) + f"_arm{a}.json").write_text(json.dumps({"protocol": protocol, "arm": a, "proj_tol_px": args.proj_tol_px,
                                                                          "seeds": recs}, indent=1) + "\n")
        n_pass = sum(1 for r in recs if r["passed"])
        print(f"arm{a}: {n_pass}/{len(recs)} seeds pass", flush=True)
    for s in sorted(per_seed):
        recs = per_seed[s]
        combined.append({"seed": s, "passed": all(recs[a]["passed"] for a in arms), "failures": [f"arm{a}:{f}" for a in arms for f in recs[a]["failures"]],
                         "arms": recs})
    # tolerance sensitivity (what-if) over all replayed cells
    cells_all = [c for s in combined for a in s["arms"].values() for c in a.get("cells", {}).values()]
    sens = {}
    if cells_all:
        for tol_abs in (8, 10, 12, 16, 24):
            sens[f"cells_within_jitter_abs{tol_abs}_count16"] = sum(1 for c in cells_all if c["replay_frame_max_abs"] <= tol_abs and max(c["replay_frame_differing_px"]) <= 16)
        for gt in (1e-9, 1e-8, 1e-7, 1e-6):
            sens[f"nocontact_cells_within_{gt:.0e}_of_baseline"] = sum(1 for c in cells_all if not c["expected_contact"] and c["baseline_state_max_dev"] <= gt)
        sens["n_cells"] = len(cells_all)
        sens["n_nocontact_cells"] = sum(1 for c in cells_all if not c["expected_contact"])
        sens["max_replay_frame_abs"] = max(c["replay_frame_max_abs"] for c in cells_all)
        sens["max_replay_differing_px"] = max(max(c["replay_frame_differing_px"]) for c in cells_all)
        sens["max_nocontact_baseline_dev"] = max(c["baseline_state_max_dev"] for c in cells_all if not c["expected_contact"])
        sens["bit_exact_cells"] = sum(1 for c in cells_all if c["replay_state_bit_exact"])
        sens["min_silhouette_patches_inlane"] = min(c["silhouette_patches"] for c in cells_all if c["kind"] and c["bbox_xyxy"] and c.get("solid") is not None and True) if cells_all else None
        sens["silhouette_patches_by_level"] = {lvl: sorted({s["arms"][a]["cells"][f"h{lvl}a0"]["silhouette_patches"] for s in combined for a in s["arms"] if "cells" in s["arms"][a]}) for lvl in (0, 1, 2, 3)}
    reasons: dict[str, int] = {}
    for r in combined:
        for f in r["failures"]:
            key = f.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    args.output.with_suffix(".json").write_text(json.dumps({"protocol": protocol, "proj_tol_px": args.proj_tol_px, "state_tol": args.state_tol,
                                                            "pixel_tolerance": {"max_abs": PIX_TOL_ABS, "max_differing_px_per_frame": PIX_TOL_COUNT}, "ghost_state_tol": GHOST_STATE_TOL,
                                                            "free_body_tol": args.free_body_tol,
                                                            "tolerance_sensitivity": sens,
                                                            "wall_s": wall, "n_seeds": len(combined), "n_pass": sum(r["passed"] for r in combined),
                                                            "failure_reasons": reasons, "seeds": combined}, indent=1) + "\n")
    msg = f"VALIDATED {sum(r['passed'] for r in combined)}/{len(combined)} seeds pass (both arms); reasons={reasons}; wall {wall:.0f}s"
    print(msg)
    for r in combined:
        for f in r["failures"]:
            if "ego_state_not_reproduced" in f or "frame_jitter" in f or "contact_label_not_replicated" in f:
                print(f"seed {r['seed']}: replay mismatch: {f}")
            elif "bbox" in f or "in_frame" in f or "projected_pose" in f:
                print(f"seed {r['seed']}: positioning gate / hazard bbox: {f}")
    try:
        args.done_log.parent.mkdir(parents=True, exist_ok=True)
        with args.done_log.open("a") as fh:
            fh.write(f"VALIDATE_DONE root={args.root} output={args.output.with_suffix('.json')} {msg} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
