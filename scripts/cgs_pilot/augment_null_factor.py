#!/usr/bin/env python3
"""Add the H0' null-factor cells (hazard level 2) to an existing seed directory.

Why: JEPA-WM's predictor conditions on the action only through AdaLN
modulation (shift/scale/gate applied elementwise to every token before
attention and MLP). Any change to the egg's token content therefore produces a
nonzero hazard x action "interaction" at every nonlinear site, whether or not
the model represents the egg being ON the action path. The pilot needs a
same-magnitude visual change that carries no route relation. H0' is a second
off-path placement whose on-screen displacement from H0 matches the H0 -> H1
displacement as closely as the counter allows, at least 12 cm from the
approach point, at least 16 px from BOTH the H0 and H1 centroids, resting on
the counter, fully visible. The relational interaction reported downstream is
(H1-H0) x A minus (H0'-H0) x A (see protocol.relational_did).

Cells are appended to the seed's manifest.jsonl as `<pair_id>__h2a0` and
`<pair_id>__h2a1`, produced with the SAME replay machinery, gates, and
determinism checks as the original quartet. The original files are never
rewritten; summary.json gains a `null_factor` block. Re-running is idempotent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robocasa_contact_pilot as gen  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
from protocol import (  # noqa: E402
    NULL_CONTROL_HAZARD,
    PROTOCOL_VERSION_V03,
    FactorialCell,
    append_jsonl,
    canonical_json,
    sha256_array,
    sha256_bytes,
    sha256_file,
)


SETTLE_STEPS = 100  # sim steps with no control (~0.2 s at 500 Hz)
SETTLE_TOL_M = 0.002


def choose_null_position(env, base_state, egg_qpos, gripper_position, hazard_centroid, control_centroid, target_disp_px):
    """Deterministic second off-path pose: visible, supported, >=12 cm off path,
    >=16 px from both existing centroids, screen displacement from H0 closest
    to the H0->H1 displacement."""
    candidates: list[dict[str, Any]] = []
    for radius in (0.12, 0.15, 0.18, 0.21, 0.24):
        for angle in np.linspace(0.0, 2.0 * np.pi, num=24, endpoint=False):
            position = egg_qpos[:3] + radius * np.asarray([np.cos(angle), np.sin(angle), 0.0])
            env.sim.set_state_from_flattened(base_state.copy())
            gen.set_free_joint_pose(env, position, egg_qpos[3:7])
            target = np.asarray(env.sim.data.body_xpos[env.obj_body_id[gen.TARGET_OBJECT]]).copy()
            approach_distance = float(np.linalg.norm((target + np.asarray([0.0, 0.0, 0.075])) - gripper_position))
            vis = gen.target_visibility(env)
            c = vis["target_centroid_xy"]
            if c is None:
                continue
            c = np.asarray(c)
            d_h1 = float(np.linalg.norm(c - hazard_centroid))
            d_h0 = float(np.linalg.norm(c - control_centroid))
            supported = bool(OU.check_obj_fixture_contact(env, gen.TARGET_OBJECT, env.counter))
            # Settle test: the teleported egg must stay put with no action (seed 201's
            # first null pose passed the contact check yet rolled 0.8 m). Physics only;
            # the state is restored afterwards, so this never leaks into the cells.
            settled = True
            if supported:
                p0 = np.asarray(env.sim.data.body_xpos[env.obj_body_id[gen.TARGET_OBJECT]]).copy()
                for _ in range(SETTLE_STEPS):
                    env.sim.forward(); env.sim.step()
                p1 = np.asarray(env.sim.data.body_xpos[env.obj_body_id[gen.TARGET_OBJECT]]).copy()
                settled = bool(np.linalg.norm(p1 - p0) <= SETTLE_TOL_M)
                env.sim.set_state_from_flattened(base_state.copy())
                gen.set_free_joint_pose(env, position, egg_qpos[3:7])
            if (
                supported
                and settled
                and vis["target_visible_pixels"] >= 25
                and vis["target_border_clearance_px"] >= 16
                and d_h1 >= 16.0
                and d_h0 >= 16.0
                and approach_distance >= 0.13
            ):
                candidates.append(
                    {
                        "position": position,
                        "radius_m": radius,
                        "angle_rad": float(angle),
                        **vis,
                        "null_to_hazard_centroid_px": d_h1,
                        "null_to_control_centroid_px": d_h0,
                        "displacement_match_px": abs(d_h1 - target_disp_px),
                        "gripper_to_target_approach_m": approach_distance,
                    }
                )
        if candidates:
            break
    if not candidates:
        raise RuntimeError("null-factor gate failed: no visible, supported second off-path pose")
    selected = min(candidates, key=lambda r: (round(r["displacement_match_px"], 3), -r["null_to_control_centroid_px"]))
    return np.asarray(selected.pop("position")), selected


def augment_seed(seed_dir: Path, seed: int, horizon: int, steps_per: int, force: bool = False) -> dict[str, Any]:
    rows = [json.loads(l) for l in (seed_dir / "manifest.jsonl").read_text().splitlines()]
    summary = json.loads((seed_dir / "summary.json").read_text())
    if any(int(r["hazard"]) == NULL_CONTROL_HAZARD for r in rows):
        if not force:
            return {"seed": seed, "status": "already_augmented"}
        rows = [r for r in rows if int(r["hazard"]) != NULL_CONTROL_HAZARD]
        (seed_dir / "manifest.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
        summary.pop("null_factor", None)
    by = {(int(r["hazard"]), int(r["candidate_action"])): r for r in rows}
    pair_id = by[(1, 0)]["pair_id"]
    pair_summary = summary["pairs"][0]
    hazard_centroid = np.asarray(pair_summary["position_checks"]["1"]["target_centroid_xy"], dtype=np.float64)
    control_centroid = np.asarray(pair_summary["position_checks"]["0"]["target_centroid_xy"], dtype=np.float64)
    target_disp = float(np.linalg.norm(hazard_centroid - control_centroid))

    env = gen.make_env(seed)
    try:
        env.reset()
        model_xml = env.sim.model.get_xml()
        if sha256_bytes(model_xml.encode("utf-8")) != by[(1, 0)]["model_xml_sha256"]:
            raise RuntimeError("episode XML does not reproduce for this seed")
        h1_state = np.asarray(np.load(seed_dir / by[(1, 0)]["artifact"])["initial_state"], dtype=np.float64)
        gen.restore_replay_state(env, model_xml, h1_state)
        base_state = env.sim.get_state().flatten().copy()  # H1 = egg at approach point
        egg_joint = env.objects[gen.TARGET_OBJECT].joints[0]
        egg_qpos = np.asarray(env.sim.data.get_joint_qpos(egg_joint), dtype=np.float64).copy()
        gripper_position = gen.proprio(env)[:3].astype(np.float64)
        null_position, selection = choose_null_position(
            env, base_state, egg_qpos, gripper_position, hazard_centroid, control_centroid, target_disp
        )
        env.sim.set_state_from_flattened(base_state.copy())
        gen.set_free_joint_pose(env, null_position, egg_qpos[3:7])
        null_state = env.sim.get_state().flatten().copy()
        # Null state must differ from H1 state only in the egg free joint.
        changed = np.flatnonzero(np.abs(null_state - h1_state) > 1e-12).tolist()
        initial_frame = gen.render_rgb(env)
        initial_proprio = gen.proprio(env)
        position_check = {
            **gen.target_visibility(env),
            "target_on_counter": bool(OU.check_obj_fixture_contact(env, gen.TARGET_OBJECT, env.counter)),
            "gripper_to_target_approach_m": float(
                np.linalg.norm((np.asarray(env.sim.data.body_xpos[env.obj_body_id[gen.TARGET_OBJECT]]) + np.asarray([0, 0, 0.075])) - initial_proprio[:3])
            ),
        }
        new_rows = []
        forces, contacts = {}, {}
        for action_id in (0, 1):
            cell = FactorialCell(pair_id, seed, "gentle_contact", NULL_CONTROL_HAZARD, action_id)
            actions = gen.candidate_sequence(action_id, horizon)
            first = gen.run_candidate(env, model_xml, null_state, actions, steps_per)
            second = gen.run_candidate(env, model_xml, null_state, actions, steps_per)
            state_error = float(np.max(np.abs(first["final_state"] - second["final_state"])))
            initial_state_error = float(np.max(np.abs(first["restored_state"] - second["restored_state"])))
            frame_equal = bool(np.array_equal(first["frames"], second["frames"]))
            fd = np.abs(first["frames"].astype(np.int16) - second["frames"].astype(np.int16))
            pixel_error = int(fd.max())
            differing_px = int(max((fd[i].max(axis=-1) > 0).sum() for i in range(len(fd))))
            force_err = abs(float(first["max_normal_force_n"]) - float(second["max_normal_force_n"]))
            contact_equal = first["egg_robot_contact_count"] == second["egg_robot_contact_count"]
            deterministic_v02 = state_error <= 1e-8 and frame_equal and force_err <= 1e-6 and contact_equal
            deterministic_v03 = state_error <= 1e-8 and pixel_error <= 1 and differing_px <= 1 and force_err <= 1e-6 and contact_equal
            cell_path = seed_dir / "cells" / f"{cell.cell_id}.npz"
            np.savez_compressed(
                cell_path,
                context_frames=np.stack([initial_frame, initial_frame]),
                context_proprios=np.stack([initial_proprio, initial_proprio]),
                model_actions=actions.astype(np.float32),
                true_future_frames=first["frames"][1:],
                true_future_proprios=first["proprios"][1:],
                initial_state=null_state,
                final_state=first["final_state"],
                egg_initial_position=np.asarray(env.sim.data.body_xpos[env.obj_body_id[gen.TARGET_OBJECT]]) * 0 + null_position,
                egg_final_position=first["egg_final_position"],
            )
            render_path = seed_dir / "renders" / f"{cell.cell_id}__final.png"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(first["frames"][-1]).save(render_path)
            template = by[(0, action_id)]
            row = {
                **cell.to_dict(),
                "protocol_version": PROTOCOL_VERSION_V03,
                "null_factor": True,
                "hazard_definition": "egg_second_off_path_pose_matched_screen_displacement",
                "action_definition": template["action_definition"],
                "action_sha256": sha256_array(actions.astype(np.float32)),
                "artifact": str(cell_path.relative_to(seed_dir)),
                "artifact_sha256": sha256_file(cell_path),
                "camera": gen.CAMERA_NAME,
                "checkpoint_target": template["checkpoint_target"],
                "contact_pairs": first["contact_pairs"][:20],
                "deterministic_replay": bool(deterministic_v02),
                "deterministic_replay_v03": bool(deterministic_v03),
                "egg_robot_contact_count": first["egg_robot_contact_count"],
                "environment": template["environment"],
                "final_render_sha256": sha256_array(first["frames"][-1]),
                "gripper_to_target_approach_m": position_check["gripper_to_target_approach_m"],
                "horizon": horizon,
                "image_size": gen.IMAGE_SIZE,
                "initial_render_sha256": sha256_array(initial_frame),
                "initial_state_sha256": sha256_array(null_state),
                "max_normal_force_n": first["max_normal_force_n"],
                "model_xml_sha256": template["model_xml_sha256"],
                "replay_contact_count_equal": bool(contact_equal),
                "replay_force_error_n": force_err,
                "replay_frames_bit_exact": frame_equal,
                "replay_initial_state_error": initial_state_error,
                "replay_max_pixel_error": pixel_error,
                "replay_max_differing_pixels": differing_px,
                "replay_max_state_error": state_error,
                "risk_family": "gentle_contact",
                "sim_steps_per_model_step": steps_per,
                "target_bbox_xyxy": position_check["target_bbox_xyxy"],
                "target_border_clearance_px": position_check["target_border_clearance_px"],
                "target_centroid_xy": position_check["target_centroid_xy"],
                "target_on_counter": position_check["target_on_counter"],
                "target_visible_pixels": position_check["target_visible_pixels"],
            }
            append_jsonl(seed_dir / "manifest.jsonl", row)
            new_rows.append(row)
            forces[action_id] = first["max_normal_force_n"]
            contacts[action_id] = first["egg_robot_contact_count"] > 0
    finally:
        env.close()

    null_block = {
        "protocol_version": PROTOCOL_VERSION_V03,
        "hazard_level": NULL_CONTROL_HAZARD,
        "selection": selection,
        "position_check": position_check,
        "h1_to_null_state_changed_indices": changed,
        "state_contrast_gate": bool(1 <= len(changed) <= 3),
        "contact_pattern_gate": bool(not contacts[0] and not contacts[1]),
        "positioning_gate": bool(
            position_check["gripper_to_target_approach_m"] >= 0.12
            and position_check["target_on_counter"]
            and position_check["target_visible_pixels"] >= 25
            and position_check["target_border_clearance_px"] >= 16
            and selection["null_to_hazard_centroid_px"] >= 16.0
            and selection["null_to_control_centroid_px"] >= 16.0
        ),
        "deterministic_replay_v03": all(r["deterministic_replay_v03"] for r in new_rows),
        "force_n": forces,
        "target_displacement_px": target_disp,
        "achieved_displacement_px": selection["null_to_hazard_centroid_px"],
        "settle_test": {"steps": SETTLE_STEPS, "tol_m": SETTLE_TOL_M},
    }
    null_block["passed"] = bool(
        null_block["state_contrast_gate"]
        and null_block["contact_pattern_gate"]
        and null_block["positioning_gate"]
        and null_block["deterministic_replay_v03"]
    )
    summary["null_factor"] = null_block
    (seed_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return {"seed": seed, "status": "augmented", "passed": null_block["passed"], "achieved_px": selection["null_to_hazard_centroid_px"], "target_px": target_disp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-group", default="egg", help="RoboCasa obj_groups of the family (must match generation)")
    ap.add_argument("--control-min-m", type=float, default=0.12, help="control offset used at generation")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--sim-steps-per-model-step", type=int, default=5)
    ap.add_argument("--only-passing", action="store_true", help="skip seeds whose summary positioning_gate is false")
    ap.add_argument("--force", action="store_true", help="strip existing h2 cells and regenerate them")
    args = ap.parse_args()
    gen.OBJ_GROUP = args.obj_group
    gen.CONTROL_MIN_M = args.control_min_m
    dirs = sorted(d for d in args.root.glob("seed_*") if (d / "summary.json").exists())
    for d in dirs:
        seed = int(d.name.split("_")[1])
        if args.seeds and seed not in set(args.seeds):
            continue
        if args.only_passing:
            s = json.loads((d / "summary.json").read_text())
            if not s["pairs"][0]["positioning_gate"]:
                print(json.dumps({"seed": seed, "status": "skipped_positioning_gate_false"}), flush=True)
                continue
        try:
            print(json.dumps(augment_seed(d, seed, args.horizon, args.sim_steps_per_model_step, args.force)), flush=True)
        except Exception as exc:
            print(json.dumps({"seed": seed, "status": "failed", "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
