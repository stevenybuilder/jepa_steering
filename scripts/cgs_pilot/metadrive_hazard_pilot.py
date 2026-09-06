#!/usr/bin/env python
"""Deterministic MetaDrive stimulus generator for the driving pedestrian/object hazard pilot (protocol v0.7 amendment).

Two modes:

``factorial``  per scene seed, 8 cells = {hazard level} x {action level}, rendered in BOTH physics arms:
    hazard levels (manifest field ``hazard``):
        0  H0    pedestrian on the right sidewalk (off-path; lateral +(w/2 + 0.5) m from the ego-lane centre)
        1  H1    pedestrian in the ego lane (on-path) at the calibrated distance
        2  H0'   matched-displacement null: the pedestrian mirrored across the ego-lane centreline
                 (lateral -(w/2 + 0.5) m, i.e. in the adjacent same-direction lane, same longitudinal, same heading).
                 Rule: |c(H0') - c(H1)| == |c(H0) - c(H1)| in pixels up to render asymmetry (the egg pilot's
                 augment_null_factor rule: null pose at the SAME screen displacement from the hazard pose as H0),
                 while staying fully in frame at 40 deg FOV (the sidewalk pose 1.5 m further out used by the spike
                 is at the frame edge). |c(H0') - c(H0)| is therefore ~2x and is reported, not gated.
        3  H1_obj  envelope-matched cone in the ego lane (on-path OBJECT), same pose as level 1
    action levels (manifest field ``candidate_action``): 0 = full brake chunk, 1 = constant throttle chunk.
    Arms: A = pedestrian solid / cone ghost; B = pedestrian ghost / cone solid (ghost = collision group 12 paired with
    Terrain only; verified in metadrive_spike.py). The context frame is identical across the two action cells of a
    hazard level and across arms; across hazard levels only the hazard pose differs.

``train``      per clip seed: random scene; 1/3 of seeds are hazard seeds rendered as BOTH a pedestrian clip and a cone
    clip at the identical pose (in-lane 50 %, right sidewalk 25 %, adjacent lane 25 %; distance U(6, 20) m at the context
    frame) so that 50 % of clips carry a hazard and the colliding-future multiset is equal across arms; random action
    chunks per model step from {brake, coast, throttle} (1/3 each), steer 0 (see TRAIN_DIST); both arms from the same
    seed. Hazard-free clips are rendered once and stored in ``shared/`` (identical in both arms by construction).

Conventions shared with the egg pilot (robocasa_contact_pilot.py / validate_hazard_labels.py):
    cells/<cell_id>.npz  context_frames [1,256,256,3] RGB uint8, true_future_frames [3,256,256,3], model_actions [3,2]
                         float32, raw_actions [9,2], initial_state / final_state (ego state vectors), ego_states [15,7]
                         per sim step (index 0 = after reset+seeding, 5 = context frame, 14 = last future frame),
                         hazard_pose, crash flags per step, distances per step.
    masks/<cell_id>.npz  hazard_mask [4,256,256] bool (semantic camera silhouette), corridor_mask [4,256,256] bool (ego-lane
                         polygon ahead projected through the camera), hazard_centroid_xy [4,2], hazard_bbox_xyxy [4,4],
                         hazard_px [4,2] projected 3D centre, footprint_mask (render difference vs hazard-free baseline,
                         incl. shadow; valid before first contact), cam intrinsics.
    manifest.jsonl       one row per cell (cell_id, pair_id, seed, hazard, candidate_action, arm, hazard_kind, solid,
                         contact metrics, visibility, replay_* fields, artifact + sha256).
Model step = 3 sim steps (env.step at 10 Hz => 0.3 s); clip = 1 context + 3 future frames.

Remote: /opt/conda/envs/metadrive/bin/python (MetaDrive 0.4.3, EGL rendering, ~1.2 GB VRAM per process).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import append_jsonl, canonical_json, sha256_array, sha256_bytes, sha256_file  # noqa: E402

PROTOCOL_VERSION_DRIVE = "cgs-metadrive-pilot-v0.7"  # rows/provenance of unchanged (v0.7/v0.8) runs keep this string
PROTOCOL_VERSION_DRIVE_V08 = PROTOCOL_VERSION_DRIVE
# v0.9 (2026-09-03; behavioral_design_v08.md s3, cross model design F1/F2/F16/F18): the stimulus factors below become
# flags. main() switches PROTOCOL_VERSION_DRIVE to this string when ANY v0.9 flag is given; without them every byte of
# the output is identical to the v0.8 generator (verified on seed 1090, d=10: all cell/mask/render/manifest files match).
PROTOCOL_VERSION_DRIVE_V09 = "cgs-metadrive-pilot-v0.9"
IMAGE_SIZE = 256
FOV_DEG = 40.0
HORIZON = 3  # model steps
SIM_STEPS_PER_MODEL_STEP = 3
PREFIX_STEPS = 5  # sim steps after reset+speed seeding before the context frame
PREFIX_THROTTLE = 0.2  # 8.0 -> 8.28 m/s over the 5 prefix steps (calibrated 2026-09-02; 0.0 decays to 7.82, 0.3 rises to 8.41)
V0_MPS = 8.0
LANE_NUM = 2
LANE_WIDTH = 3.5
SIDEWALK_EXTRA_M = 0.5  # sidewalk pose = outer-lane edge + 0.5 m (lateral 2.25 m; the spike's 2.75 m leaves the frame at d <= 8 m)
HAZARD_DIST_M = 9.0  # centre-to-centre longitudinal gap at the context frame; calibrated 2026-09-02 (immovable body: contact at future sim step 7 = model step 3 under throttle for seeds 0-2; brake min gap 3.1-3.9 m; d=8 straddles steps 5/6, d=10 hits the last step, d=7 pushes the sidewalk pose out of frame)
ACTION_CHUNKS = {0: (0.0, -1.0), 1: (0.0, 0.5)}  # A0 brake, A1 throttle
ACTION_NAMES = {0: "brake", 1: "throttle"}
HAZARD_LEVELS = {0: ("ped", "sidewalk_right"), 1: ("ped", "in_lane"), 2: ("ped", "adjacent_lane_left"), 3: ("cone", "in_lane")}
HAZARD_NAMES = {0: "H0_ped_sidewalk", 1: "H1_ped_in_lane", 2: "H0p_ped_mirror_null", 3: "H1_obj_cone_in_lane"}
ARMS = {"A": {"ped": True, "cone": False}, "B": {"ped": False, "cone": True}}  # solid flags per hazard kind
MIN_BBOX_PX = 48  # FOOTPRINT bbox (render difference vs the hazard-free baseline, |diff| > 8, incl. shadow) as in the spike; reported
DIFF_THRESH = 8
# Visibility gate (coordinator decision 2026-09-02, preregistered): the exact silhouette must cover >= MIN_PATCHES DINOv3
# ViT-L/16 patches (16x16 px on the 256 px frame, 16x16 grid) in the context frame of the in-lane cells.
MIN_PATCHES = 6
PATCH_PX = 16
SEM_TOL = 40  # L-inf tolerance on the semantic colour
CONE_SEMANTIC = "TRAFFIC_SIGN"  # re-tag the cone so its mask is unambiguous (no signs on S/C blocks)
GHOST_GROUP = 12
# Solid-body model. Re-measured 2026-09-02 (_dbg_contact): panda3d `setStatic(True)` (once or twice) leaves a 70 kg body
# that the car PENETRATES (transient 10.5 -> 9.5 m/s, min centre distance 0.01-0.3 m); `setMass(0)` (Bullet static,
# infinite mass) stops the car at the body (min centre distance 2.5 m = half ego length + radius, 9.4 -> 3.7 -> 0.5 m/s);
# a free 70 kg body is knocked over (z 0.875 -> 0.31) and the ego barely slows (9.4 -> 9.0 m/s).
HAZARD_BODY = "static"  # static (setMass 0, immovable) | dynamic (70 kg free body)
# v0.9 stimulus factors (all opt-in via flags; defaults reproduce v0.8 byte for byte):
LATERAL_OFFSET_M = 0.0  # --lateral-offset: in-lane levels 1 and 3 only, + = right of the lane centre; H0/H0' poses unchanged
SETTLE_STEPS = 0  # --settle-steps: physics-only steps (decision_repeat Bullet ticks each, no control) after spawning, before the
#   prefix, applied to EVERY episode of the run (baselines too) so all cells share the same physics history. Measured 2026-09-03
#   (settle_probe): free 70 kg in-lane pedestrian drifts 0; free cone drifts 1.9e-5 m then Bullet puts it to sleep at ~10 steps
#   (drift exactly 0 afterwards); the sidewalk pedestrian slides 0.75 m along the raised kerb and rests by step 30.
RANDOMIZE: dict[str, tuple[float, float]] | None = None  # --randomize "dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5"
RANDOM_FACTORS = ("dist", "prefix", "lateral")  # hazard_dist_m, prefix_throttle, in-lane lateral offset (m)
V09_ACTIVE = False  # set by main() when any v0.9 flag is given: protocol string bump + extra metadata keys
# train-mode preregistered distribution
TRAIN_DIST = {
    # kind_paired (default, protocol v0.7 amendment P1 "collision-future counts equal across arms"): every hazard seed is
    # rendered as BOTH a pedestrian clip and a cone clip at the identical pose/actions, so the set of colliding futures is
    # the same multiset in both arms by construction (arm A collides on the ped copies, arm B on the cone copies).
    # p_hazard = 1/3 of seeds then gives 50 % hazard clips (2 per hazard seed vs 1 per hazard-free seed).
    "kind_paired": True,
    "p_hazard": 1 / 3,
    "p_kind": {"ped": 0.5, "cone": 0.5},  # used only when kind_paired is False
    "p_pose": {"in_lane": 0.5, "sidewalk_right": 0.25, "adjacent_lane_left": 0.25},
    "dist_m": [6.0, 20.0],  # off-path poses; in-lane poses use inlane_dist_m (collision-support bias, coordinator 2026-09-02)
    "inlane_dist_m": [6.0, 14.0],
    "v0_mps": [6.0, 10.0],
    "p_action": {"brake": 0.25, "coast": 0.25, "throttle": 0.5},
    "throttle_brake": {"brake": -1.0, "coast": 0.0, "throttle": 0.5},
    "steer_sigma": 0.0,  # 2026-09-02 pilot: steer noise (sigma 0.05) amplifies Bullet's 1e-13 summation-order differences
    # (hazard body present vs absent, ghost vs solid) to 1e-8..1e-6 m in the ego state, which shows as texture speckle
    # (thousands of pixels at <= 93/255) in otherwise identical pre-contact frames -> arms no longer pixel-identical.
    # With steer = 0 the deviation stays <= 1e-9 (factorial pilot). Enable with --steer-sigma for a steering variant.
    "steer": "clip(N(0, steer_sigma), -0.15, 0.15) per model step (steer_sigma = 0 -> exactly 0)",
    "map": "block_sequence 'S' + 2 blocks drawn uniformly from {S, C}; ego spawns in the outer lane of the fixed first block",
}
STATE_FIELDS = ["x", "y", "z", "heading", "vx", "vy", "speed"]
FRAME_MEANING = ["context", "future1", "future2", "future3"]

os.environ.setdefault("PYTHONHASHSEED", "0")


# ----------------------------------------------------------------------------- env


def scene_map_config(seed: int, mode: str) -> str:
    """First block after the spawn block is always straight (hazard region); the next two are seed-random {S, C}."""
    rng = np.random.RandomState(seed * 7919 + (0 if mode == "factorial" else 1_000_003))
    return "S" + "".join(rng.choice(["S", "C"]) for _ in range(2))


def make_env(num_scenarios: int = 200_000):
    from metadrive.envs import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.sensors.semantic_camera import SemanticCamera
    from metadrive.component.pgblock.first_block import FirstPGBlock

    cfg = dict(
        use_render=False,
        image_observation=True,
        norm_pixel=False,
        stack_size=1,
        traffic_density=0.0,
        accident_prob=0.0,
        num_scenarios=num_scenarios,
        start_seed=0,
        store_map=False,
        random_spawn_lane_index=False,
        random_lane_width=False,
        random_lane_num=False,
        random_agent_model=False,
        map_config=dict(type="block_sequence", config="SSS", lane_num=LANE_NUM, lane_width=LANE_WIDTH),
        crash_human_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        out_of_road_done=False,
        horizon=1000,
        show_interface=False,
        show_logo=False,
        show_fps=False,
        log_level=logging.WARNING,
        map_region_size=256,
        agent_configs={"default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, LANE_NUM - 1)}},
        sensors={"rgb_camera": (RGBCamera, IMAGE_SIZE, IMAGE_SIZE), "semantic_camera": (SemanticCamera, IMAGE_SIZE, IMAGE_SIZE)},
        vehicle_config={"image_source": "rgb_camera"},
    )
    env = MetaDriveEnv(cfg)
    return env


def scenario_count(seeds) -> int:
    """MetaDrive asserts seed < num_scenarios; keep the pilot's 200_000 floor so seeds below it see an identical env."""
    return max(200_000, int(max(seeds)) + 1)


def reset_scene(env, seed: int, map_cfg: str):
    """Hard reset onto the scene map for `seed`; align cameras; set the 40 deg FOV on both lenses."""
    env.config["map_config"]["config"] = map_cfg
    if env.engine is not None:
        env.engine.global_config["map_config"]["config"] = map_cfg
    obs, _ = env.reset(seed=seed)
    rgb = env.engine.get_sensor("rgb_camera")
    sem = env.engine.get_sensor("semantic_camera")
    rgb.get_lens().setFov(FOV_DEG)
    sem.get_lens().setFov(FOV_DEG)
    sem.track(rgb.cam.getParent(), rgb.cam.getPos(), rgb.cam.getHpr())
    return obs


def frame_rgb(obs) -> np.ndarray:
    img = obs["image"]
    if img.ndim == 4:
        img = img[..., -1]
    return np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB


def semantic_rgb(env) -> np.ndarray:
    img = env.engine.get_sensor("semantic_camera").perceive(False)
    return np.ascontiguousarray(np.asarray(img)[..., ::-1])


def semantic_colour(label: str) -> np.ndarray:
    from metadrive.constants import Semantics

    return np.asarray(getattr(Semantics, label)[1], dtype=np.int16)


def colour_mask(sem_rgb: np.ndarray, colour: np.ndarray) -> np.ndarray:
    return np.abs(sem_rgb.astype(np.int16) - colour[None, None]).max(-1) <= SEM_TOL


def ego_state(env) -> np.ndarray:
    a = env.agent
    v = a.velocity
    return np.array([a.position[0], a.position[1], a.origin.getZ(), a.heading_theta, v[0], v[1], float(a.speed)], dtype=np.float64)


def ego_lane(env):
    return env.agent.lane if env.agent.lane is not None else env.agent.navigation.current_ref_lanes[-1]


def lane_pose(env, s_ahead: float, lateral: float) -> tuple[np.ndarray, float]:
    """World (x, y) and heading at `s_ahead` m along the ego lane from the ego's current longitudinal, `lateral` m
    to the right (+) / left (-). The spawn lane is straight and collinear with the next two roads (FirstPGBlock 30 m
    + block 'S' >= 40 m), so straight-line extrapolation beyond lane.length is exact for < 70 m."""
    lane = ego_lane(env)
    long_, _ = lane.local_coordinates(env.agent.position)
    pos = np.asarray(lane.position(long_ + s_ahead, lateral), dtype=np.float64)
    return pos, float(lane.heading_theta_at(long_ + s_ahead))


def sidewalk_lateral(env) -> float:
    return ego_lane(env).width / 2 + SIDEWALK_EXTRA_M


def level_lateral(env, where: str, inlane_lateral: float | None = None) -> float:
    """Lateral offset (m, + = right) of a hazard placement. In-lane placements take the v0.9 lateral offset (default
    LATERAL_OFFSET_M = 0.0 = v0.8); the sidewalk / mirror poses are unchanged by it."""
    if inlane_lateral is None:
        inlane_lateral = LATERAL_OFFSET_M
    return {"in_lane": inlane_lateral, "sidewalk_right": sidewalk_lateral(env), "adjacent_lane_left": -sidewalk_lateral(env)}[where]


def pose_for_level(env, level: int, dist: float) -> tuple[np.ndarray, float, float, str]:
    kind, where = HAZARD_LEVELS[level]
    lat = level_lateral(env, where)
    pos, heading = lane_pose(env, dist, lat)
    return pos, heading, lat, kind


def settle_physics(env, n_steps: int) -> None:
    """Step ONLY the Bullet world (no vehicle control, no manager step) for n_steps x decision_repeat ticks."""
    ticks = int(n_steps) * int(env.config["decision_repeat"])
    for _ in range(ticks):
        env.engine.step_physics_world()


# ----------------------------------------------------------------------------- v0.9 factor helpers


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats by None (JSON null) and unwrap numpy scalars. Used only on the generator's
    summary writing path: canonical_json (allow_nan=False) raised on a NaN control-pose centroid distance (F16)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_randomize(spec: str | None) -> dict[str, tuple[float, float]] | None:
    """'dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5' -> {factor: (lo, hi)}; factors not listed keep the fixed flags."""
    if not spec:
        return None
    out: dict[str, tuple[float, float]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, rng = item.partition(":")
        name = name.strip()
        if name not in RANDOM_FACTORS:
            raise ValueError(f"unknown randomize factor {name!r}; choose from {RANDOM_FACTORS}")
        lo, hi = (float(x) for x in rng.split(","))
        if hi < lo:
            raise ValueError(f"randomize {name}: hi < lo ({lo}, {hi})")
        out[name] = (lo, hi)
    return out or None


def factor_rng(seed: int, name: str) -> np.random.RandomState:
    """Generator seeded by (seed, factor name) only: reproducible, platform independent, identical across arms."""
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode()).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "big"))


def draw_factors(seed: int, spec: dict[str, tuple[float, float]] | None) -> dict[str, float] | None:
    if not spec:
        return None
    return {name: float(lo + factor_rng(seed, name).uniform() * (hi - lo)) for name, (lo, hi) in spec.items()}


def seed_factors(seed: int, dist: float) -> tuple[float, float, float, dict[str, float] | None]:
    """(hazard_dist_m, prefix_throttle, in-lane lateral, drawn) for one seed: drawn values override the fixed flags."""
    drawn = draw_factors(seed, RANDOMIZE)
    d = drawn or {}
    return float(d.get("dist", dist)), float(d.get("prefix", PREFIX_THROTTLE)), float(d.get("lateral", LATERAL_OFFSET_M)), drawn


def v09_metadata(pt: float, lat_inlane: float, drawn: dict[str, float] | None) -> dict[str, Any]:
    return {"prefix_throttle": pt, "hazard_lateral_offset_m": lat_inlane, "fov_deg": FOV_DEG, "settle_steps": SETTLE_STEPS,
            "randomized_factors": drawn, "randomize_spec": ({k: list(v) for k, v in RANDOMIZE.items()} if RANDOMIZE else None)}


def apply_row_factors(row: dict[str, Any]) -> tuple[float, float, float]:
    """Set the module factors (FOV, settle steps, body, prefix throttle) from a saved manifest row and return
    (hazard_dist_m, prefix_throttle, in-lane lateral). v0.8 rows carry none of the v0.9 keys -> v0.8 values."""
    global FOV_DEG, SETTLE_STEPS, HAZARD_BODY, PREFIX_THROTTLE
    HAZARD_BODY = str(row.get("hazard_body", HAZARD_BODY))
    FOV_DEG = float(row.get("fov_deg", 40.0))
    SETTLE_STEPS = int(row.get("settle_steps", 0))
    PREFIX_THROTTLE = float(row.get("prefix_throttle", 0.2))
    return float(row["hazard_dist_m"]), PREFIX_THROTTLE, float(row.get("hazard_lateral_offset_m", 0.0))


_CONE_CLS = None


def cone_class():
    global _CONE_CLS
    if _CONE_CLS is None:
        from metadrive_spike import matched_cone_class
        from metadrive.constants import Semantics

        base = matched_cone_class()

        class MatchedConeSem(base):
            SEMANTIC_LABEL = getattr(Semantics, CONE_SEMANTIC).label

        _CONE_CLS = MatchedConeSem
    return _CONE_CLS


def spawn_hazard_at(env, kind: str, pos: np.ndarray, heading: float, solid: bool):
    """Spawn + body model (see HAZARD_BODY); ghost = collision group 12 paired only with Terrain (metadrive_spike.py:
    render untouched, no contactTest hit, no crash flag, no physics response)."""
    from panda3d.core import BitMask32
    from metadrive.constants import CollisionGroup, CameraTagStateKey
    from metadrive.component.traffic_participants.pedestrian import Pedestrian

    cls = Pedestrian if kind == "ped" else cone_class()
    obj = env.engine.spawn_object(cls, position=(float(pos[0]), float(pos[1])), heading_theta=float(heading), force_spawn=True)
    if HAZARD_BODY == "static":
        obj.body.setMass(0.0)
    elif HAZARD_BODY != "dynamic":
        raise ValueError(HAZARD_BODY)
    if kind == "cone":
        obj.origin.setTag(CameraTagStateKey.Semantic, cls.SEMANTIC_LABEL)
    if not solid:
        world = env.engine.physics_world.dynamic_world
        obj.body.setIntoCollideMask(BitMask32.bit(GHOST_GROUP))
        for other in range(32):
            world.setGroupCollisionFlag(GHOST_GROUP, other, other == CollisionGroup.Terrain.getLowestOnBit())
    return obj


def pose_pedestrian_animation():
    """Pose the (shared) pedestrian model explicitly so frames never depend on wall-clock animation time."""
    from metadrive.component.traffic_participants.pedestrian import Pedestrian

    for model in Pedestrian._MODEL.values():
        ac = model.get_anim_control("Take 001")
        ac.stop()
        ac.pose(1)


def project(env, xyz) -> tuple[float, float, bool]:
    from panda3d.core import Point2, Point3

    rgb = env.engine.get_sensor("rgb_camera")
    p = rgb.cam.getRelativePoint(env.engine.render, Point3(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    p2 = Point2()
    ok = bool(rgb.get_lens().project(p, p2)) and p[1] > 0
    return (p2[0] + 1) / 2 * IMAGE_SIZE, (1 - p2[1]) / 2 * IMAGE_SIZE, ok


def camera_intrinsics(env) -> dict[str, Any]:
    rgb = env.engine.get_sensor("rgb_camera")
    lens = rgb.get_lens()
    return {"fov": [float(v) for v in lens.getFov()], "cam_pos_vehicle": [float(v) for v in rgb.cam.getPos()],
            "cam_hpr_vehicle": [float(v) for v in rgb.cam.getHpr()], "near_far": [float(lens.getNear()), float(lens.getFar())]}


def corridor_mask(env, s_max: float = 60.0, s_min: float = 1.5) -> np.ndarray:
    """Ego-lane polygon from s_min..s_max m ahead projected into the image (cv2 clips out-of-frame vertices)."""
    import cv2

    lane = ego_lane(env)
    long_, _ = lane.local_coordinates(env.agent.position)
    w = lane.width / 2
    ss = np.linspace(s_min, s_max, 24)
    left = [project(env, (*lane.position(long_ + s, -w), 0.0))[:2] for s in ss]
    right = [project(env, (*lane.position(long_ + s, +w), 0.0))[:2] for s in ss]
    poly = np.asarray(left + right[::-1], dtype=np.float64)
    poly = np.clip(poly, -4096, 4096)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def hazard_gap(env, obj, radius: float) -> tuple[float, float, float, float]:
    """Centre distance, longitudinal / lateral offsets in the ego frame, and box-to-cylinder gap (m)."""
    a = env.agent
    d = np.asarray(obj.position, dtype=np.float64) - np.asarray(a.position, dtype=np.float64)
    h = float(a.heading_theta)
    lon = d[0] * np.cos(h) + d[1] * np.sin(h)
    lat = -d[0] * np.sin(h) + d[1] * np.cos(h)
    dx = max(0.0, abs(lon) - a.LENGTH / 2 - radius)
    dy = max(0.0, abs(lat) - a.WIDTH / 2 - radius)
    return float(np.hypot(d[0], d[1])), float(lon), float(lat), float(np.hypot(dx, dy))


# ----------------------------------------------------------------------------- episode


def raw_actions_from_chunks(chunks: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(chunks, dtype=np.float64), SIM_STEPS_PER_MODEL_STEP, axis=0)


def run_episode(env, seed: int, map_cfg: str, hazard: dict | None, chunks: np.ndarray, v0: float = V0_MPS,
                prefix_throttle: float | None = None, with_masks: bool = True, baseline_frames: np.ndarray | None = None) -> dict[str, Any]:
    """Reset, seed the ego speed, spawn the hazard (optional), run PREFIX_STEPS then HORIZON model steps.

    hazard = {"kind": ped|cone, "s_ahead": m (measured from the ego position at the CONTEXT frame), "lateral": m,
              "solid": bool}. The context ego longitudinal is known from the hazard-free prefix travel of the scene
              (``prefix_travel``), so the object is spawned at reset at s_ahead + prefix_travel.
    prefix_throttle None -> the module value PREFIX_THROTTLE (resolved at call time so --prefix-throttle / row values apply).
    SETTLE_STEPS > 0 (v0.9): reset -> spawn -> physics-only settle -> seed the ego speed; ``hazard_pose`` is the settled
    (context) pose and ``hazard_spawn_pose`` / ``hazard_settle_drift_m`` record the spawn pose and how far it moved.
    """
    from metadrive.component.traffic_participants.pedestrian import Pedestrian

    if prefix_throttle is None:
        prefix_throttle = PREFIX_THROTTLE
    reset_scene(env, seed, map_cfg)
    if SETTLE_STEPS <= 0:
        env.agent.set_velocity([1.0, 0.0], float(v0), in_local_frame=True)
    obj = None
    radius = 0.0
    spawn_pose = None
    if hazard is not None:
        pos, heading = lane_pose(env, hazard["s_ahead"] + hazard["prefix_travel"], hazard["lateral"])
        obj = spawn_hazard_at(env, hazard["kind"], pos, heading, hazard["solid"])
        radius = Pedestrian.RADIUS
        spawn_pose = [float(obj.position[0]), float(obj.position[1]), float(obj.origin.getZ()), float(obj.heading_theta)]
    if SETTLE_STEPS > 0:
        settle_physics(env, SETTLE_STEPS)
        env.agent.set_velocity([1.0, 0.0], float(v0), in_local_frame=True)
    pose_pedestrian_animation()
    raw = raw_actions_from_chunks(chunks)
    n_steps = PREFIX_STEPS + raw.shape[0]
    states = [ego_state(env)]
    frames: list[np.ndarray] = []
    sem_frames: list[np.ndarray] = []
    corridors: list[np.ndarray] = []
    hazard_px: list[list[float]] = []
    crash_human, crash_object, info_crash, dists, lons, lats, gaps, obj_z = [], [], [], [], [], [], [], []
    obj_pose0 = None
    if obj is not None:
        obj_pose0 = [float(obj.position[0]), float(obj.position[1]), float(obj.origin.getZ()), float(obj.heading_theta)]
    frame_steps = {PREFIX_STEPS + k * SIM_STEPS_PER_MODEL_STEP for k in range(HORIZON + 1)}
    for t in range(n_steps):
        act = [0.0, prefix_throttle] if t < PREFIX_STEPS else [float(raw[t - PREFIX_STEPS, 0]), float(raw[t - PREFIX_STEPS, 1])]
        pose_pedestrian_animation()
        obs, _, _, _, info = env.step(act)
        states.append(ego_state(env))
        a = env.agent
        crash_human.append(bool(a.crash_human))
        crash_object.append(bool(a.crash_object))
        info_crash.append(bool(info.get("crash_human", False) or info.get("crash_object", False)))
        if obj is not None:
            dc, lon, lat, gap = hazard_gap(env, obj, radius)
            dists.append(dc); lons.append(lon); lats.append(lat); gaps.append(gap); obj_z.append(float(obj.origin.getZ()))
        if (t + 1) in frame_steps:
            frames.append(frame_rgb(obs).copy())
            if with_masks:
                sem_frames.append(semantic_rgb(env).copy())
                corridors.append(corridor_mask(env))
                if obj is not None:
                    u, v, ok = project(env, (obj.position[0], obj.position[1], obj.origin.getZ()))
                    hazard_px.append([u, v, float(ok)])
    out: dict[str, Any] = {
        "frames": np.stack(frames), "states": np.stack(states), "raw_actions": raw.astype(np.float64),
        "chunks": np.asarray(chunks, dtype=np.float64), "crash_human": np.asarray(crash_human), "crash_object": np.asarray(crash_object),
        "info_crash": np.asarray(info_crash), "camera": camera_intrinsics(env), "map_cfg": map_cfg,
        "ego_dims": [float(env.agent.LENGTH), float(env.agent.WIDTH)],
    }
    if with_masks:
        out["sem_frames"] = np.stack(sem_frames)
        out["corridor_mask"] = np.stack(corridors)
    if obj is not None:
        out.update({
            "hazard_pose": obj_pose0, "hazard_radius": radius, "center_dist": np.asarray(dists), "lon": np.asarray(lons),
            "lat": np.asarray(lats), "gap": np.asarray(gaps), "hazard_z": np.asarray(obj_z),
            "hazard_px": np.asarray(hazard_px, dtype=np.float64) if with_masks else None,
            "hazard_final_pose": [float(obj.position[0]), float(obj.position[1]), float(obj.origin.getZ()), float(obj.heading_theta)],
            "hazard_spawn_pose": spawn_pose,
            "hazard_settle_drift_m": float(np.linalg.norm(np.asarray(obj_pose0[:3]) - np.asarray(spawn_pose[:3]))),
        })
        if with_masks:
            colour = semantic_colour("PEDESTRIAN" if hazard["kind"] == "ped" else CONE_SEMANTIC)
            out["hazard_mask"] = np.stack([colour_mask(s, colour) for s in sem_frames])
        if baseline_frames is not None:
            fd = np.abs(out["frames"].astype(np.int16) - baseline_frames.astype(np.int16)).max(-1)
            out["footprint_mask"] = fd > DIFF_THRESH
            fc = first_true(out["crash_human"][PREFIX_STEPS:] | out["crash_object"][PREFIX_STEPS:])
            # footprint valid only while the ego trajectory equals the baseline (before the first contact)
            out["footprint_valid"] = np.asarray([fc is None or (k * SIM_STEPS_PER_MODEL_STEP) <= fc for k in range(HORIZON + 1)])
        env.engine.clear_objects([obj.id], force_destroy=True)  # pooled reuse re-applies setStatic/ghost mask -> divergent physics
    return out


def first_true(flags: np.ndarray) -> int | None:
    idx = np.flatnonzero(np.asarray(flags))
    return int(idx[0]) if idx.size else None


def patch_count(mask: np.ndarray, patch: int = PATCH_PX) -> int:
    """Number of DINOv3 patches (16x16 px cells of the 256 px frame) with at least one mask pixel."""
    m = np.asarray(mask, dtype=bool)
    g = m.shape[0] // patch
    return int(m[: g * patch, : g * patch].reshape(g, patch, g, patch).any(axis=(1, 3)).sum())


def bbox_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"target_visible_pixels": 0, "target_bbox_xyxy": None, "target_centroid_xy": None, "target_border_clearance_px": -1,
                "bbox_w": 0, "bbox_h": 0, "patches": 0}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {"target_visible_pixels": int(mask.sum()), "target_bbox_xyxy": bbox, "patches": patch_count(mask),
            "target_centroid_xy": [float(xs.mean()), float(ys.mean())],
            "target_border_clearance_px": int(min(bbox[0], bbox[1], IMAGE_SIZE - 1 - bbox[2], IMAGE_SIZE - 1 - bbox[3])),
            "bbox_w": bbox[2] - bbox[0] + 1, "bbox_h": bbox[3] - bbox[1] + 1}


def prefix_travel_of(states: np.ndarray) -> float:
    return float(np.hypot(*(states[PREFIX_STEPS, :2] - states[0, :2])))


def contact_summary(ep: dict[str, Any]) -> dict[str, Any]:
    n_pre = PREFIX_STEPS
    ch = ep["crash_human"][n_pre:]
    co = ep["crash_object"][n_pre:]
    any_c = ch | co
    fc = first_true(any_c)
    res = {
        "crash_human_any": bool(ch.any()), "crash_object_any": bool(co.any()),
        "crash_prefix_any": bool((ep["crash_human"][:n_pre] | ep["crash_object"][:n_pre]).any()),
        "first_contact_step": fc, "first_contact_model_step": (fc // SIM_STEPS_PER_MODEL_STEP + 1) if fc is not None else None,
        "egg_robot_contact_count": int(any_c.sum()),
        "context_speed_mps": float(ep["states"][n_pre, 6]), "final_speed_mps": float(ep["states"][-1, 6]),
    }
    if "gap" in ep:
        res.update({"min_center_distance_m": float(ep["center_dist"].min()), "min_gap_m": float(ep["gap"].min()),
                    "argmin_gap_step": int(np.argmin(ep["gap"])) - n_pre, "context_center_distance_m": float(ep["center_dist"][n_pre - 1]),
                    "context_longitudinal_gap_m": float(ep["lon"][n_pre - 1]), "context_lateral_m": float(ep["lat"][n_pre - 1]),
                    "hazard_z_min": float(ep["hazard_z"].min()), "hazard_z_max": float(ep["hazard_z"].max())})
    return res


# ----------------------------------------------------------------------------- factorial


def factorial_seed(env, seed: int, out_dir: Path, arms: list[str], dist: float, two_pass: bool = True,
                   prefix_throttle: float | None = None, lateral: float | None = None, drawn: dict[str, float] | None = None) -> dict[str, Any]:
    """Generate the 8 cells x arms for one scene seed; returns the per-seed summary (also written per arm).

    two_pass: write ``manifest_pending.jsonl``; a FRESH process (``factorial_replay``) then replays the identical
    episode sequence, fills the replay_* fields and writes ``manifest.jsonl`` (bit-exact physics is only reproducible
    across processes with the same in-process history; a second run inside this process differs post-contact by ~1e-8).
    v0.9: ``prefix_throttle`` (ego speed seeding; None -> PREFIX_THROTTLE), ``lateral`` (in-lane offset, m; None ->
    LATERAL_OFFSET_M), ``drawn`` (the per-seed --randomize draws, recorded in metadata; identical across arms).
    """
    pt = PREFIX_THROTTLE if prefix_throttle is None else float(prefix_throttle)
    lat_inlane = LATERAL_OFFSET_M if lateral is None else float(lateral)
    map_cfg = scene_map_config(seed, "factorial")
    pair_id = f"drive_seed{seed:06d}"
    chunks = {a: np.repeat(np.asarray([ACTION_CHUNKS[a]], dtype=np.float64), HORIZON, axis=0) for a in (0, 1)}
    # hazard-free baselines (define the context ego position and the NONE anchor)
    base = {a: run_episode(env, seed, map_cfg, None, chunks[a], prefix_throttle=pt) for a in (0, 1)}
    prefix_travel = prefix_travel_of(base[0]["states"])
    if abs(prefix_travel - prefix_travel_of(base[1]["states"])) > 1e-9:
        raise RuntimeError("baseline prefix travel differs between action chunks")
    sem0 = base[0]["sem_frames"][0]
    for lab in ("PEDESTRIAN", CONE_SEMANTIC):
        if colour_mask(sem0, semantic_colour(lab)).any():
            raise RuntimeError(f"hazard-free context frame contains semantic colour {lab}")
    summaries = {}
    # The context frame is rendered before any action or arm difference (bit-exact physics state); the EGL renderer
    # jitters by <= ~10/255 on a handful of pixels between renders, so the first render of each hazard level (arm order,
    # a=0) is stored as THE context frame of all four (arm, action) cells of that level. The per-cell render jitter
    # against it is recorded (context_render_jitter_px / _max_abs).
    shared_ctx: dict[int, np.ndarray] = {}
    for arm in arms:
        adir = out_dir / f"arm{arm}" / f"seed_{seed}"
        (adir / "cells").mkdir(parents=True, exist_ok=True)
        (adir / "masks").mkdir(parents=True, exist_ok=True)
        (adir / "renders").mkdir(parents=True, exist_ok=True)
        manifest = adir / ("manifest_pending.jsonl" if two_pass else "manifest.jsonl")
        for old in (adir / "manifest_pending.jsonl", adir / "manifest.jsonl"):
            if old.exists():
                old.unlink()
        np.savez_compressed(adir / "scene_baseline.npz",
                            frames_a0=base[0]["frames"], frames_a1=base[1]["frames"], states_a0=base[0]["states"], states_a1=base[1]["states"],
                            corridor_mask=base[0]["corridor_mask"], prefix_travel=np.float64(prefix_travel), map_cfg=np.array(map_cfg))
        rows = []
        vis = {}
        contacts = {}
        for level in (0, 1, 2, 3):
            kind, where = HAZARD_LEVELS[level]
            solid = ARMS[arm][kind]
            lat = level_lateral(env, where, lat_inlane)
            hz = {"kind": kind, "s_ahead": dist, "lateral": lat, "solid": solid, "prefix_travel": prefix_travel}
            for a in (0, 1):
                cell_id = f"{pair_id}__h{level}a{a}"
                ep = run_episode(env, seed, map_cfg, hz, chunks[a], prefix_throttle=pt, baseline_frames=base[a]["frames"])
                cs = contact_summary(ep)
                if level not in shared_ctx:
                    shared_ctx[level] = ep["frames"][0].copy()
                ctx_jit = np.abs(ep["frames"][0].astype(np.int16) - shared_ctx[level].astype(np.int16))
                ctx_jitter = {"context_render_jitter_px": int((ctx_jit.max(-1) > 0).sum()), "context_render_jitter_max_abs": int(ctx_jit.max()),
                              "context_frame_source": f"arm{arms[0]}_h{level}a0"}
                ep["frames"][0] = shared_ctx[level]
                v = bbox_stats(ep["hazard_mask"][0])
                fp = bbox_stats(ep["footprint_mask"][0])
                v.update({"footprint_bbox_xyxy": fp["target_bbox_xyxy"], "footprint_wh": [fp["bbox_w"], fp["bbox_h"]],
                          "footprint_pixels": fp["target_visible_pixels"], "footprint_border_clearance_px": fp["target_border_clearance_px"],
                          "footprint_centroid_xy": fp["target_centroid_xy"]})
                vis[(level, a)] = v
                contacts[(level, a)] = cs
                hp = ep["hazard_px"]
                cell_path = adir / "cells" / f"{cell_id}.npz"
                np.savez_compressed(
                    cell_path,
                    context_frames=ep["frames"][:1], true_future_frames=ep["frames"][1:],
                    model_actions=ep["chunks"].astype(np.float32), raw_actions=ep["raw_actions"].astype(np.float32),
                    initial_state=ep["states"][0], context_state=ep["states"][PREFIX_STEPS], final_state=ep["states"][-1],
                    ego_states=ep["states"], state_fields=np.array(STATE_FIELDS),
                    crash_human=ep["crash_human"], crash_object=ep["crash_object"], info_crash=ep["info_crash"],
                    center_dist=ep["center_dist"], gap=ep["gap"], lon=ep["lon"], lat=ep["lat"], hazard_z=ep["hazard_z"],
                    hazard_pose=np.asarray(ep["hazard_pose"]), hazard_final_pose=np.asarray(ep["hazard_final_pose"]),
                    hazard_kind=np.array(kind), solid=np.bool_(solid), arm=np.array(arm), map_cfg=np.array(map_cfg),
                    prefix_travel=np.float64(prefix_travel),
                )
                per_frame_stats = [bbox_stats(m) for m in ep["hazard_mask"]]
                np.savez_compressed(
                    adir / "masks" / f"{cell_id}.npz",
                    hazard_mask=ep["hazard_mask"], corridor_mask=ep["corridor_mask"],
                    hazard_centroid_xy=np.asarray([st["target_centroid_xy"] or [np.nan, np.nan] for st in per_frame_stats], dtype=np.float32),
                    hazard_bbox_xyxy=np.asarray([st["target_bbox_xyxy"] or [-1, -1, -1, -1] for st in per_frame_stats], dtype=np.int32),
                    hazard_px=hp[:, :2].astype(np.float32), hazard_px_valid=hp[:, 2].astype(bool),
                    footprint_mask=ep["footprint_mask"], footprint_valid=ep["footprint_valid"],
                    frame_index_meaning=np.array(FRAME_MEANING), fov=np.float64(FOV_DEG),
                    cam_pos=np.asarray(ep["camera"]["cam_pos_vehicle"]), cam_hpr=np.asarray(ep["camera"]["cam_hpr_vehicle"]),
                )
                try:
                    from PIL import Image

                    Image.fromarray(np.concatenate([ep["frames"][0], ep["frames"][-1]], axis=1)).save(adir / "renders" / f"{cell_id}__ctx_final.png")
                except Exception:  # pragma: no cover
                    pass
                row = {
                    "protocol_version": PROTOCOL_VERSION_DRIVE, "pair_id": pair_id, "seed": seed, "risk_family": "pedestrian_collision",
                    "hazard": level, "candidate_action": a, "cell_id": cell_id, "arm": arm, "hazard_kind": kind, "solid": solid,
                    "hazard_definition": HAZARD_NAMES[level], "action_definition": ACTION_NAMES[a],
                    "environment": f"MetaDrive-0.4.3/block_sequence({map_cfg},lane_num={LANE_NUM},lane_width={LANE_WIDTH})",
                    "camera": f"rgb_camera(vehicle,(0,0.8,1.5),fov={FOV_DEG:g})", "image_size": IMAGE_SIZE, "horizon": HORIZON,
                    "sim_steps_per_model_step": SIM_STEPS_PER_MODEL_STEP, "prefix_steps": PREFIX_STEPS, "prefix_throttle": pt,
                    "v0_mps": V0_MPS, "hazard_body": HAZARD_BODY, "hazard_dist_m": dist, "hazard_lateral_m": lat, "hazard_pose_xyzh": ep["hazard_pose"],
                    "hazard_final_pose_xyzh": ep["hazard_final_pose"], "prefix_travel_m": prefix_travel, "map_cfg": map_cfg,
                    **{k: v[k] for k in ("target_visible_pixels", "target_bbox_xyxy", "target_centroid_xy", "target_border_clearance_px")},
                    "bbox_wh": [v["bbox_w"], v["bbox_h"]], "silhouette_patches": v["patches"], "footprint_bbox_xyxy": v["footprint_bbox_xyxy"], "footprint_wh": v["footprint_wh"],
                    "footprint_pixels": v["footprint_pixels"], "footprint_border_clearance_px": v["footprint_border_clearance_px"],
                    "hazard_px_context": [float(hp[0, 0]), float(hp[0, 1])],
                    "hazard_px_all": hp[:, :2].tolist(),
                    **cs,
                    "hazard_centroid_xy": v["target_centroid_xy"], "hazard_bbox_xyxy": v["target_bbox_xyxy"], **ctx_jitter,
                    "min_distance_m": cs["min_gap_m"], "contact": bool(cs["egg_robot_contact_count"] > 0), "contact_step": cs["first_contact_step"],
                    "crash_human": cs["crash_human_any"], "crash_object": cs["crash_object_any"],
                    "max_normal_force_n": 0.0,  # not available from Bullet contactTest; contact = flag
                    "contact_pairs": [["ego", kind]] if cs["egg_robot_contact_count"] else [],
                    "artifact": str(cell_path.relative_to(adir)), "artifact_sha256": sha256_file(cell_path),
                    "initial_state_sha256": sha256_array(ep["states"][0]), "action_sha256": sha256_array(ep["raw_actions"]),
                    "initial_render_sha256": sha256_array(ep["frames"][0]), "final_render_sha256": sha256_array(ep["frames"][-1]),
                }
                if V09_ACTIVE:  # new keys only under v0.9 so unchanged runs stay byte-identical
                    row.update(v09_metadata(pt, lat_inlane, drawn))
                    row.update({"hazard_spawn_pose_xyzh": ep["hazard_spawn_pose"], "hazard_settle_drift_m": ep["hazard_settle_drift_m"]})
                append_jsonl(manifest, row)
                rows.append(row)
        # scene-level gates
        ctx = {k: r["initial_render_sha256"] for k, r in zip([(l, a) for l in (0, 1, 2, 3) for a in (0, 1)], rows)}
        same_ctx_across_actions = all(ctx[(l, 0)] == ctx[(l, 1)] for l in (0, 1, 2, 3))
        expected = {(l, a): (a == 1 and ((l == 1 and ARMS[arm]["ped"]) or (l == 3 and ARMS[arm]["cone"]))) for l in (0, 1, 2, 3) for a in (0, 1)}
        pattern_ok = all((contacts[k]["egg_robot_contact_count"] > 0) == expected[k] for k in expected)
        c0, c1, c2 = (np.asarray(vis[(l, 0)]["target_centroid_xy"] or [np.nan, np.nan]) for l in (0, 1, 2))
        d_h0 = float(np.linalg.norm(c0 - c1)); d_h0p = float(np.linalg.norm(c2 - c1)); d_null_to_h0 = float(np.linalg.norm(c2 - c0))
        null_match = bool(np.isfinite(d_h0) and np.isfinite(d_h0p) and abs(d_h0p - d_h0) <= 0.10 * max(d_h0, 1e-9))
        gate_bbox = all(vis[(l, 0)]["footprint_wh"][0] >= MIN_BBOX_PX and vis[(l, 0)]["footprint_wh"][1] >= MIN_BBOX_PX for l in (1, 3))
        silhouette_48 = all(vis[(l, 0)]["bbox_w"] >= MIN_BBOX_PX and vis[(l, 0)]["bbox_h"] >= MIN_BBOX_PX for l in (1, 3))
        gate_patches = all(vis[(l, 0)]["patches"] >= MIN_PATCHES for l in (1, 3))
        in_frame = all(vis[(l, 0)]["target_visible_pixels"] > 0 and vis[(l, 0)]["target_border_clearance_px"] >= 1 for l in (0, 1, 2, 3))
        # ghost / off-path cells must track the hazard-free baseline
        base_dev = {}
        for (l, a), r in zip(expected, rows):
            st = np.load(adir / r["artifact"])["ego_states"]
            base_dev[f"h{l}a{a}"] = float(np.abs(st - base[a]["states"]).max())
        nocontact_dev = max(v for k, v in base_dev.items() if not expected[(int(k[1]), int(k[3]))])
        dmin = {k: contacts[k]["min_gap_m"] for k in contacts}
        did_ped = (dmin[(1, 1)] - dmin[(1, 0)]) - (dmin[(0, 1)] - dmin[(0, 0)])
        did_obj = (dmin[(3, 1)] - dmin[(3, 0)]) - (dmin[(0, 1)] - dmin[(0, 0)])
        solid_lvl = 1 if ARMS[arm]["ped"] else 3
        # interaction gate: the solid in-lane throttle cell reaches the body (gap ~0); every brake cell and every solid
        # off-path throttle cell keeps >= 0.5 m (the ghost in-lane throttle cell passes THROUGH the ghost, gap 0 by design)
        ghost_lvl = 3 if solid_lvl == 1 else 1
        others = [v for k, v in dmin.items() if k != (solid_lvl, 1) and k != (ghost_lvl, 1)]
        min_dist_gate = bool(dmin[(solid_lvl, 1)] <= 0.05 and min(others) >= 0.5)
        if not (gate_patches and in_frame):
            sil = [[vis[(l, 0)]["bbox_w"], vis[(l, 0)]["bbox_h"]] for l in (1, 3)]
            pat = [vis[(l, 0)]["patches"] for l in (1, 3)]
            print(f"seed {seed} arm{arm}: positioning gate failed: hazard bbox silhouette {sil} patches {pat} (need >= {MIN_PATCHES}); in_frame={in_frame}", flush=True)
        summary = {
            "pair_id": pair_id, "seed": seed, "arm": arm, "map_cfg": map_cfg, "hazard_dist_m": dist, "prefix_travel_m": prefix_travel,
            "min_distance_did_m": float(did_ped), "min_distance_did_object_m": float(did_obj), "min_distance_interaction_gate": min_dist_gate,
            "min_distance_by_cell": {f"h{l}a{a}": dmin[(l, a)] for (l, a) in dmin},
            "context_speed_mps": contacts[(1, 0)]["context_speed_mps"],
            "contacts": {f"h{l}a{a}": contacts[(l, a)] for (l, a) in contacts},
            "position_checks": {str(l): vis[(l, 0)] for l in (0, 1, 2, 3)},
            "control_pose_selection": {"hazard_to_control_centroid_px": d_h0, "null_to_hazard_centroid_px": d_h0p,
                                       "null_to_control_centroid_px": d_null_to_h0, "rule": "mirror of H0 across the ego-lane centreline"},
            "baseline_state_max_dev": base_dev, "nocontact_cells_max_dev_from_baseline": nocontact_dev,
            "positioning_gate": bool(gate_patches and in_frame), "silhouette_patches_gate": gate_patches, "min_patches": MIN_PATCHES,
            "footprint_bbox_48_gate": gate_bbox, "silhouette_bbox_48_gate": silhouette_48, "in_frame_gate": in_frame,
            "same_context_across_actions": same_ctx_across_actions, "contact_pattern_gate": pattern_ok,
            "null_displacement_match_gate": null_match, "ghost_nocontact_gate": bool(nocontact_dev <= 1e-9),
            "state_contrast_gate": True, "force_interaction_gate": pattern_ok,
            "contact_did": float(paired_flag_did(contacts, ARMS[arm])),
            "force_did_n": 0.0,
            "all_replays_deterministic": None if two_pass else False,  # filled by factorial_replay
        }
        if V09_ACTIVE:
            summary["stimulus_factors"] = {"hazard_dist_m": dist, **v09_metadata(pt, lat_inlane, drawn), "hazard_body": HAZARD_BODY}
            summary["hazard_settle_drift_m"] = {f"h{l}a{a}": r["hazard_settle_drift_m"] for (l, a), r in zip(expected, rows)}
        summ = {"provenance": provenance(), "pairs": [summary],
                "phase0_stimulus_gate": {"complete_octet": len(rows) == 8, "positioning": summary["positioning_gate"],
                                         "intended_contact_pattern": pattern_ok, "same_context_across_actions": same_ctx_across_actions,
                                         "null_displacement_match": null_match, "ghost_nocontact": summary["ghost_nocontact_gate"],
                                         "deterministic_replay": summary["all_replays_deterministic"]}}
        # json_safe: non-finite floats (e.g. the control-pose centroid distance when a pose leaves the frame) -> null
        (adir / "summary.json").write_text(canonical_json(json_safe(summ)) + "\n", encoding="utf-8")
        summaries[arm] = summary
    return summaries


def any_contacts(ep: dict[str, Any]) -> int:
    return int((ep["crash_human"][PREFIX_STEPS:] | ep["crash_object"][PREFIX_STEPS:]).sum())


def paired_flag_did(contacts: dict, arm_solid: dict) -> float:
    lvl = 1 if arm_solid["ped"] else 3
    c = lambda l, a: float(contacts[(l, a)]["egg_robot_contact_count"] > 0)
    return (c(lvl, 1) - c(lvl, 0)) - (c(0, 1) - c(0, 0))


def provenance() -> dict[str, Any]:
    import metadrive

    prov = {"protocol_version": PROTOCOL_VERSION_DRIVE, "generator": Path(__file__).name, "generator_sha256": sha256_file(__file__),
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "metadrive": getattr(metadrive, "__version__", "0.4.3"), "image_size": IMAGE_SIZE, "fov_deg": FOV_DEG, "horizon": HORIZON,
            "sim_steps_per_model_step": SIM_STEPS_PER_MODEL_STEP, "prefix_steps": PREFIX_STEPS, "prefix_throttle": PREFIX_THROTTLE,
            "v0_mps": V0_MPS, "action_chunks": {str(k): list(v) for k, v in ACTION_CHUNKS.items()}, "hazard_levels": HAZARD_NAMES,
            "arms": ARMS, "hazard_body": HAZARD_BODY + (" (setMass(0): immovable, car stops at the body)" if HAZARD_BODY == "static" else " (70 kg free body, knocked over)"),
            "ghost": "collision group 12 paired with Terrain only",
            "null_rule": HAZARD_LEVELS[2], "train_distribution": TRAIN_DIST}
    if V09_ACTIVE:
        prov["v09_stimulus"] = {"lateral_offset_m": LATERAL_OFFSET_M, "settle_steps": SETTLE_STEPS, "fov_deg": FOV_DEG,
                                "prefix_throttle": PREFIX_THROTTLE,
                                "randomize_spec": ({k: list(v) for k, v in RANDOMIZE.items()} if RANDOMIZE else None),
                                "randomize_rule": "per seed and factor: U(lo, hi) from RandomState(sha256('<seed>:<factor>')[:4]); identical across arms",
                                "settle_rule": "reset -> spawn -> settle_steps x decision_repeat Bullet ticks (no control) -> seed ego speed; all episodes incl. baselines",
                                "camera": f"rgb_camera(vehicle,(0,0.8,1.5),fov={FOV_DEG:g})"}
    return prov


def factorial_replay_seed(env, seed: int, out_dir: Path, arms: list[str]) -> dict[str, Any]:
    """Second pass in a FRESH process: replay the identical episode sequence of ``factorial_seed`` (baselines, then
    every cell in order), compare with the saved cells, add the replay_* fields and write manifest.jsonl."""
    pending = {a: out_dir / f"arm{a}" / f"seed_{seed}" / "manifest_pending.jsonl" for a in arms}
    rows = {a: [json.loads(l) for l in pending[a].read_text().splitlines() if l.strip()] for a in arms}
    map_cfg = str(rows[arms[0]][0]["map_cfg"])
    # per-seed factors come from the pending rows (v0.9 draws / flags; v0.8 rows -> v0.8 values), never from this process's flags
    dist, pt, lat_inlane = apply_row_factors(rows[arms[0]][0])
    chunks = {a: np.repeat(np.asarray([ACTION_CHUNKS[a]], dtype=np.float64), HORIZON, axis=0) for a in (0, 1)}
    base = {a: run_episode(env, seed, map_cfg, None, chunks[a], prefix_throttle=pt) for a in (0, 1)}
    prefix_travel = prefix_travel_of(base[0]["states"])
    report: dict[str, Any] = {"seed": seed, "cells": {}}
    for arm in arms:
        adir = out_dir / f"arm{arm}" / f"seed_{seed}"
        new_rows = []
        for level in (0, 1, 2, 3):
            kind, where = HAZARD_LEVELS[level]
            solid = ARMS[arm][kind]
            lat = level_lateral(env, where, lat_inlane)
            hz = {"kind": kind, "s_ahead": dist, "lateral": lat, "solid": solid, "prefix_travel": prefix_travel}
            for a in (0, 1):
                row = next(r for r in rows[arm] if int(r["hazard"]) == level and int(r["candidate_action"]) == a)
                ep = run_episode(env, seed, map_cfg, hz, chunks[a], prefix_throttle=pt, baseline_frames=base[a]["frames"])
                cell = np.load(adir / row["artifact"])
                saved_states = np.asarray(cell["ego_states"], dtype=np.float64)
                saved_frames = np.concatenate([cell["context_frames"], cell["true_future_frames"]])
                fd = np.abs(ep["frames"].astype(np.int16) - saved_frames.astype(np.int16))
                per_frame = (fd.max(-1) > 0).reshape(fd.shape[0], -1).sum(1)
                n_contact = any_contacts(ep)
                row = dict(row)
                row.update({
                    "replay_max_state_error": float(np.abs(ep["states"] - saved_states).max()),
                    "replay_initial_state_error": float(np.abs(ep["states"][0] - saved_states[0]).max()),
                    "replay_states_bit_exact": bool(np.array_equal(ep["states"], saved_states)),
                    "replay_frames_bit_exact": bool(np.array_equal(ep["frames"], saved_frames)),
                    "replay_max_pixel_error": int(fd.max()), "replay_max_differing_pixels_per_frame": int(per_frame.max()),
                    "replay_differing_pixels_per_frame": per_frame.tolist(),
                    "replay_force_error_n": 0.0,
                    "replay_contact_count_equal": bool(n_contact == int(row["egg_robot_contact_count"])),
                    "replay_min_gap_error_m": float(abs(ep["gap"].min() - float(row["min_gap_m"]))),
                    "replay_process": "fresh_process_same_history",
                })
                row["deterministic_replay"] = bool(row["replay_max_state_error"] <= 1e-8 and row["replay_contact_count_equal"]
                                                   and row["replay_max_pixel_error"] <= 8 and row["replay_max_differing_pixels_per_frame"] <= 16)
                new_rows.append(row)
                report["cells"][f"arm{arm}_{row['cell_id']}"] = {k: row[k] for k in row if k.startswith("replay_") or k == "deterministic_replay"}
        manifest = adir / "manifest.jsonl"
        if manifest.exists():
            manifest.unlink()
        for r in new_rows:
            append_jsonl(manifest, r)
        summ = json.loads((adir / "summary.json").read_text())
        det = all(r["deterministic_replay"] for r in new_rows)
        if not det:
            bad = [(r["cell_id"], r["replay_max_state_error"], r["replay_max_pixel_error"], r["replay_max_differing_pixels_per_frame"]) for r in new_rows if not r["deterministic_replay"]]
            print(f"seed {seed} arm{arm}: replay mismatch in {len(bad)} cells (cell_id, state_err, max_px, n_px): {bad}", flush=True)
        summ["pairs"][0]["all_replays_deterministic"] = det
        summ["pairs"][0]["replay_states_bit_exact"] = all(r["replay_states_bit_exact"] for r in new_rows)
        summ["phase0_stimulus_gate"]["deterministic_replay"] = det
        (adir / "summary.json").write_text(canonical_json(json_safe(summ)) + "\n", encoding="utf-8")
        (adir / "replay_report.json").write_text(json.dumps(report, indent=1) + "\n")
        pending[arm].unlink()
    return report


# ----------------------------------------------------------------------------- train


def sample_train_clip(seed: int) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    spec: dict[str, Any] = {"seed": seed, "map_cfg": scene_map_config(seed, "train"), "v0": float(rng.uniform(*TRAIN_DIST["v0_mps"]))}
    spec["has_hazard"] = bool(rng.rand() < TRAIN_DIST["p_hazard"])
    if spec["has_hazard"]:
        kinds, pk = zip(*TRAIN_DIST["p_kind"].items())
        spec["kind"] = str(kinds[rng.choice(len(kinds), p=pk)])  # drawn for RNG-stream stability; ignored when kind_paired
        spec["kinds"] = ["ped", "cone"] if TRAIN_DIST["kind_paired"] else [spec["kind"]]
        poses, pp = zip(*TRAIN_DIST["p_pose"].items())
        spec["pose"] = str(poses[rng.choice(len(poses), p=pp)])
        u = rng.rand()  # one uniform draw; mapped to the pose-specific range so the RNG stream is pose-independent
        lo, hi = TRAIN_DIST["inlane_dist_m"] if spec["pose"] == "in_lane" else TRAIN_DIST["dist_m"]
        spec["dist"] = float(lo + u * (hi - lo))
    names, pa = zip(*TRAIN_DIST["p_action"].items())
    cls = [str(names[rng.choice(len(names), p=pa)]) for _ in range(HORIZON)]
    steer = np.clip(rng.normal(0.0, 1.0, size=HORIZON) * TRAIN_DIST["steer_sigma"], -0.15, 0.15)  # same RNG stream for any sigma
    spec["action_classes"] = cls
    spec["chunks"] = np.stack([steer, [TRAIN_DIST["throttle_brake"][c] for c in cls]], axis=1).astype(np.float64).tolist()
    return spec


def train_range(env, seeds: list[int], out_dir: Path, tag: str) -> dict[str, Any]:
    shards = {"armA": [], "armB": [], "shared": []}
    t0 = time.time()
    n_runs = 0
    for seed in seeds:
        spec = sample_train_clip(seed)
        chunks = np.asarray(spec["chunks"], dtype=np.float64)
        map_cfg = spec["map_cfg"]
        if not spec["has_hazard"]:
            ep = run_episode(env, seed, map_cfg, None, chunks, v0=spec["v0"], with_masks=False)
            n_runs += 1
            shards["shared"].append(pack_clip(ep, spec, "shared", None))
            continue
        base = run_episode(env, seed, map_cfg, None, chunks, v0=spec["v0"], with_masks=False)
        n_runs += 1
        pt = prefix_travel_of(base["states"])
        lat = {"in_lane": 0.0, "sidewalk_right": sidewalk_lateral(env), "adjacent_lane_left": -sidewalk_lateral(env)}[spec["pose"]]
        for kind in spec["kinds"]:
            kspec = dict(spec, kind=kind)
            for arm in ("A", "B"):
                hz = {"kind": kind, "s_ahead": spec["dist"], "lateral": lat, "solid": ARMS[arm][kind], "prefix_travel": pt}
                ep = run_episode(env, seed, map_cfg, hz, chunks, v0=spec["v0"], with_masks=True, baseline_frames=base["frames"])
                n_runs += 1
                shards[f"arm{arm}"].append(pack_clip(ep, kspec, arm, base))
    elapsed = time.time() - t0
    written = {}
    for name, clips in shards.items():
        if not clips:
            continue
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"shard_{tag}.npz"
        arrays = {k: np.stack([c[k] for c in clips]) for k in clips[0] if isinstance(clips[0][k], np.ndarray)}
        np.savez_compressed(path, **arrays)
        with (d / f"shard_{tag}.jsonl").open("w") as fh:
            for c in clips:
                fh.write(canonical_json(c["meta"]) + "\n")
        written[name] = {"path": str(path), "n": len(clips), "bytes": path.stat().st_size}
    stats = {"seeds": [seeds[0], seeds[-1]], "n_clips": len(seeds), "n_runs": n_runs, "elapsed_s": elapsed,
             "clips_per_s": len(seeds) / elapsed, "runs_per_s": n_runs / elapsed, "written": written}
    (out_dir / f"stats_{tag}.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


def pack_clip(ep: dict[str, Any], spec: dict[str, Any], arm: str, base: dict[str, Any] | None) -> dict[str, Any]:
    cs = contact_summary(ep)
    suffix = f"_{spec['kind']}" if (spec["has_hazard"] and TRAIN_DIST["kind_paired"]) else ""
    meta = {"clip_id": f"drive_train{spec['seed']:07d}{suffix}", "seed": spec["seed"], "arm": arm, "map_cfg": spec["map_cfg"], "v0_mps": spec["v0"],
            "kind_paired": bool(TRAIN_DIST["kind_paired"]),
            "has_hazard": spec["has_hazard"], "hazard_kind": spec.get("kind"), "hazard_pose": spec.get("pose"), "hazard_dist_m": spec.get("dist"),
            "hazard_level": ({"in_lane": {"ped": 1, "cone": 3}, "sidewalk_right": {"ped": 0, "cone": 0}, "adjacent_lane_left": {"ped": 2, "cone": 2}}
                             [spec["pose"]][spec["kind"]] if spec["has_hazard"] else -1),
            "solid": (ARMS[arm][spec["kind"]] if spec["has_hazard"] and arm in ARMS else None),
            "action_classes": spec["action_classes"], "model_actions": spec["chunks"], "contact": cs["egg_robot_contact_count"] > 0,
            "first_contact_step": cs["first_contact_step"], "first_contact_model_step": cs["first_contact_model_step"],
            "min_gap_m": cs.get("min_gap_m"), "min_center_distance_m": cs.get("min_center_distance_m"),
            "context_speed_mps": cs["context_speed_mps"], "hazard_pose_xyzh": ep.get("hazard_pose"),
            "frames_sha256": sha256_array(ep["frames"])}
    if base is not None:
        fd = np.abs(ep["frames"].astype(np.int16) - base["frames"].astype(np.int16))
        meta["baseline_state_max_dev"] = float(np.abs(ep["states"] - base["states"]).max())
        meta["baseline_frame_diff_px_per_frame"] = (fd.max(-1) > 0).reshape(fd.shape[0], -1).sum(1).tolist()
    packed = {"frames": ep["frames"], "actions": ep["chunks"].astype(np.float32), "raw_actions": ep["raw_actions"].astype(np.float32),
              "ego_states": ep["states"].astype(np.float64), "crash": (ep["crash_human"] | ep["crash_object"]), "meta": meta}
    if "hazard_mask" in ep:
        packed["hazard_mask_packed"] = np.packbits(ep["hazard_mask"], axis=-1)
        packed["footprint_mask_packed"] = np.packbits(ep["footprint_mask"], axis=-1)
        packed["footprint_valid"] = ep["footprint_valid"]
        packed["hazard_px"] = ep["hazard_px"][:, :2].astype(np.float32)
        bs = bbox_stats(ep["hazard_mask"][0])
        meta["bbox_wh_context"] = [bs["bbox_w"], bs["bbox_h"]]
        meta["silhouette_patches_context"] = bs["patches"]
        meta["footprint_wh_context"] = [bbox_stats(ep["footprint_mask"][0])["bbox_w"], bbox_stats(ep["footprint_mask"][0])["bbox_h"]]
    return packed


def load_train_arm(root: Path, arm: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Reader: arm-specific shards + shared hazard-free shards, sorted by seed."""
    metas, arrays = [], []
    for d in (root / f"arm{arm}", root / "shared"):
        for p in sorted(d.glob("shard_*.npz")):
            z = np.load(p)
            ms = [json.loads(l) for l in p.with_suffix(".jsonl").read_text().splitlines() if l.strip()]
            for i, m in enumerate(ms):
                metas.append(m)
                arrays.append({k: z[k][i] for k in ("frames", "actions", "raw_actions", "ego_states", "crash")})
    order = np.argsort([m["seed"] for m in metas])
    out = {k: np.stack([arrays[i][k] for i in order]) for k in ("frames", "actions", "raw_actions", "ego_states", "crash")}
    return out, [metas[i] for i in order]


# ----------------------------------------------------------------------------- calibration


def calibrate(out_dir: Path, seeds: list[int], dists: list[float], prefix_throttles: list[float]) -> dict[str, Any]:
    """Prefix throttle -> context speed; hazard distance -> contact step per (kind, action, solid); env timings; history test."""
    global PREFIX_THROTTLE
    t0 = time.time()
    env = make_env()
    res: dict[str, Any] = {"env_create_s": time.time() - t0, "prefix": [], "sweep": [], "resets": []}
    chunks = {a: np.repeat(np.asarray([ACTION_CHUNKS[a]], dtype=np.float64), HORIZON, axis=0) for a in (0, 1)}
    for pt in prefix_throttles:
        ep = run_episode(env, seeds[0], scene_map_config(seeds[0], "factorial"), None, chunks[1], prefix_throttle=pt, with_masks=False)
        res["prefix"].append({"prefix_throttle": pt, "context_speed": float(ep["states"][PREFIX_STEPS, 6]),
                              "speed_profile": ep["states"][:, 6].round(3).tolist(), "prefix_travel": prefix_travel_of(ep["states"])})
    for seed in seeds:
        map_cfg = scene_map_config(seed, "factorial")
        t1 = time.time()
        base = run_episode(env, seed, map_cfg, None, chunks[1], with_masks=False)
        res["resets"].append({"seed": seed, "map_cfg": map_cfg, "episode_s": time.time() - t1, "lane_len": float(ego_lane(env).length)})
        pt = prefix_travel_of(base["states"])
        base0 = run_episode(env, seed, map_cfg, None, chunks[0], with_masks=False)
        for d in dists:
            for level in (0, 1, 2, 3):
                kind, where = HAZARD_LEVELS[level]
                lat = {"in_lane": 0.0, "sidewalk_right": sidewalk_lateral(env), "adjacent_lane_left": -sidewalk_lateral(env)}[where]
                for a in ((0, 1) if where == "in_lane" else (0,)):
                    hz = {"kind": kind, "s_ahead": d, "lateral": lat, "solid": True, "prefix_travel": pt}
                    t1 = time.time()
                    ep = run_episode(env, seed, map_cfg, hz, chunks[a], with_masks=(a == 0), baseline_frames=(base0 if a == 0 else base)["frames"])
                    cs = contact_summary(ep)
                    row = {"seed": seed, "dist": d, "level": level, "kind": kind, "action": a, "episode_s": time.time() - t1, **cs}
                    if a == 0:
                        row["bbox_ctx"] = bbox_stats(ep["hazard_mask"][0])
                        row["footprint_ctx"] = bbox_stats(ep["footprint_mask"][0])
                        row["hazard_px_ctx"] = ep["hazard_px"][0].tolist()
                    res["sweep"].append(row)
    env.close()
    (out_dir / "calibration.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def history_worker(out: Path, seeds: list[int]) -> None:
    env = make_env()
    chunks = np.repeat(np.asarray([ACTION_CHUNKS[1]], dtype=np.float64), HORIZON, axis=0)
    rec = {}
    for s in seeds:
        m = scene_map_config(s, "factorial")
        base = run_episode(env, s, m, None, chunks, with_masks=False)
        hz = {"kind": "ped", "s_ahead": HAZARD_DIST_M, "lateral": 0.0, "solid": True, "prefix_travel": prefix_travel_of(base["states"])}
        ep = run_episode(env, s, m, hz, chunks, with_masks=False)
        ep2 = run_episode(env, s, m, hz, chunks, with_masks=False)
        rec[f"s{s}_base"] = base["states"]; rec[f"s{s}_ep"] = ep["states"]; rec[f"s{s}_ep2"] = ep2["states"]
        rec[f"s{s}_fr"] = ep["frames"]; rec[f"s{s}_fr2"] = ep2["frames"]
    np.savez_compressed(out, **rec)
    env.close()


def history_test(out_dir: Path) -> dict[str, Any]:
    """Does the physics depend on the in-process episode history? P1 runs seeds [0,1]; P2 runs [1]; P3 runs [1] again."""
    paths = [out_dir / f"hist_{i}.npz" for i in range(3)]
    for p, seeds in zip(paths, ([0, 1], [1], [1])):
        cp = subprocess.run([sys.executable, __file__, "history_worker", "--out", str(out_dir), "--path", str(p), "--seeds", *map(str, seeds)],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr[-3000:])
    a, b, c = (np.load(p) for p in paths)
    def sd(x, y): return float(np.abs(x - y).max())
    def fdiff(x, y):
        d = np.abs(x.astype(np.int16) - y.astype(np.int16)); return {"max": int(d.max()), "npx": int((d.max(-1) > 0).sum())}
    res = {
        "seed1_ep_P1(after seed0)_vs_P2(fresh)": sd(a["s1_ep"], b["s1_ep"]), "seed1_ep_P2_vs_P3(fresh,fresh)": sd(b["s1_ep"], c["s1_ep"]),
        "seed1_ep_vs_ep2_within_P2": sd(b["s1_ep"], b["s1_ep2"]), "seed1_ep2_P2_vs_P3": sd(b["s1_ep2"], c["s1_ep2"]),
        "seed1_base_P1_vs_P2": sd(a["s1_base"], b["s1_base"]),
        "frames_seed1_ep_P2_vs_P3": fdiff(b["s1_fr"], c["s1_fr"]), "frames_seed1_ep_vs_ep2_within_P2": fdiff(b["s1_fr"], b["s1_fr2"]),
        "frames_seed1_ep_P1_vs_P2": fdiff(a["s1_fr"], b["s1_fr"]),
    }
    (out_dir / "history_test.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


# ----------------------------------------------------------------------------- drivers


def chunk_list(items: list[int], n: int) -> list[list[int]]:
    return [items[i::n] for i in range(n) if items[i::n]]


def spawn_workers(cmds: list[list[str]], procs: int, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    running: list[tuple[subprocess.Popen, Any]] = []
    idx = 0
    while idx < len(cmds) or running:
        while idx < len(cmds) and len(running) < procs:
            lf = (log_dir / f"worker_{idx:04d}.log").open("w")
            running.append((subprocess.Popen(cmds[idx], stdout=lf, stderr=subprocess.STDOUT), lf))
            idx += 1
        time.sleep(1.0)
        still = []
        for p, lf in running:
            if p.poll() is None:
                still.append((p, lf))
            else:
                lf.close()
                if p.returncode != 0:
                    print(f"worker failed rc={p.returncode}: {' '.join(p.args)}", flush=True)
        running = still


def main() -> None:
    global HAZARD_BODY, PREFIX_THROTTLE, LATERAL_OFFSET_M, FOV_DEG, SETTLE_STEPS, RANDOMIZE, V09_ACTIVE, PROTOCOL_VERSION_DRIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["factorial", "train", "calibrate", "history", "history_worker", "factorial_worker", "factorial_replay", "train_worker"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seed-range", type=int, nargs=2, default=None, help="start end (exclusive)")
    ap.add_argument("--arms", default="AB")
    ap.add_argument("--hazard-dist", type=float, default=HAZARD_DIST_M)
    ap.add_argument("--hazard-body", choices=["static", "dynamic"], default=HAZARD_BODY)
    ap.add_argument("--steer-sigma", type=float, default=TRAIN_DIST["steer_sigma"], help="train: steer noise sigma per model step (0 = none)")
    ap.add_argument("--no-kind-pairing", action="store_true", help="train: one hazard kind per hazard seed (p_hazard 0.5) instead of ped+cone copies (p_hazard 1/3)")
    ap.add_argument("--inlane-dist", type=float, nargs=2, default=TRAIN_DIST["inlane_dist_m"], help="train: in-lane hazard distance range (m) at the context frame")
    ap.add_argument("--p-throttle", type=float, default=TRAIN_DIST["p_action"]["throttle"], help="train: P(throttle) per model step; brake/coast share the rest equally")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seeds-per-proc", type=int, default=1, help="factorial: seeds per worker process (1 = fresh process per seed, required for bit-exact validator replay; 0 = one chunk per proc)")
    ap.add_argument("--no-replay-check", action="store_true")
    ap.add_argument("--clips-per-shard", type=int, default=100)
    ap.add_argument("--path", type=Path)
    ap.add_argument("--dists", type=float, nargs="*", default=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0])
    ap.add_argument("--prefix-throttles", type=float, nargs="*", default=[0.0, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--tag", default=None)
    # v0.9 stimulus flags (factorial modes). Giving ANY of them (even at its default value) marks the run v0.9:
    # protocol string cgs-metadrive-pilot-v0.9 and the extra metadata keys; with none, output is byte-identical to v0.8.
    v09 = ap.add_argument_group("v0.9 stimulus factors (opt-in)")
    v09.add_argument("--prefix-throttle", type=float, default=None, help=f"ego throttle over the {PREFIX_STEPS} prefix steps = context speed (v0.8: {PREFIX_THROTTLE}; 0.0 -> 7.82, 0.5 -> ~8.6 m/s)")
    v09.add_argument("--lateral-offset", type=float, default=None, help="in-lane hazard lateral offset (m, + = right) for levels 1 and 3 only (v0.8: 0.0); H0/H0' unchanged")
    v09.add_argument("--randomize", type=str, default=None, help="per-seed uniform draws, e.g. 'dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5'; each factor's RNG is seeded by (seed, factor) so draws are reproducible and identical across arms; unlisted factors keep the fixed flags")
    v09.add_argument("--fov", type=float, default=None, help="camera horizontal FOV in degrees for both lenses (v0.8: 40); changes hazard pixel size -> validator silhouette gate (>= 6 patches) must still hold")
    v09.add_argument("--settle-steps", type=int, default=None, help="physics-only steps (decision_repeat ticks each) after spawning before the prefix, all episodes (v0.8: 0); 30 lets a free body (--hazard-body dynamic) fall asleep")
    args = ap.parse_args()
    HAZARD_BODY = args.hazard_body
    v09_flags = {k: getattr(args, k) for k in ("prefix_throttle", "lateral_offset", "randomize", "fov", "settle_steps") if getattr(args, k) is not None}
    if v09_flags:
        V09_ACTIVE = True
        PROTOCOL_VERSION_DRIVE = PROTOCOL_VERSION_DRIVE_V09
        if args.prefix_throttle is not None:
            PREFIX_THROTTLE = float(args.prefix_throttle)
        if args.lateral_offset is not None:
            LATERAL_OFFSET_M = float(args.lateral_offset)
        if args.fov is not None:
            FOV_DEG = float(args.fov)
        if args.settle_steps is not None:
            SETTLE_STEPS = int(args.settle_steps)
        RANDOMIZE = parse_randomize(args.randomize)
    v09_argv = [f"--{k.replace('_', '-')}={v}" for k, v in v09_flags.items()]  # passed through to worker / replay processes
    TRAIN_DIST["steer_sigma"] = float(args.steer_sigma)
    TRAIN_DIST["inlane_dist_m"] = [float(args.inlane_dist[0]), float(args.inlane_dist[1])]
    pt = float(args.p_throttle)
    TRAIN_DIST["p_action"] = {"brake": (1 - pt) / 2, "coast": (1 - pt) / 2, "throttle": pt}
    if args.no_kind_pairing:
        TRAIN_DIST["kind_paired"] = False
        TRAIN_DIST["p_hazard"] = 0.5
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds or [])
    if args.seed_range:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))

    if args.mode == "calibrate":
        res = calibrate(args.out, seeds or [0, 1, 2], args.dists, args.prefix_throttles)
        print(json.dumps(res, indent=1, default=str)[:20000])
        return
    if args.mode == "history_worker":
        history_worker(args.path, seeds)
        return
    if args.mode == "history":
        print(json.dumps(history_test(args.out), indent=1))
        return
    if args.mode == "factorial_worker":
        env = make_env(num_scenarios=scenario_count(seeds))
        t0 = time.time()
        for seed in seeds:
            t1 = time.time()
            dist, pt, lat, drawn = seed_factors(seed, args.hazard_dist)
            s = factorial_seed(env, seed, args.out, list(args.arms), dist, not args.no_replay_check, prefix_throttle=pt, lateral=lat, drawn=drawn)
            line = {"seed": seed, "elapsed_s": time.time() - t1, **{a: {k: v for k, v in s[a].items() if k.endswith("_gate")} for a in s}}
            if V09_ACTIVE:
                line["factors"] = {"hazard_dist_m": dist, "prefix_throttle": pt, "lateral_offset_m": lat, "drawn": drawn}
            print(json.dumps(json_safe(line)), flush=True)
        print(json.dumps({"seeds": len(seeds), "total_s": time.time() - t0, "seeds_per_s": len(seeds) / (time.time() - t0)}), flush=True)
        env.close()
        return
    if args.mode == "factorial_replay":
        env = make_env(num_scenarios=scenario_count(seeds))
        for seed in seeds:
            t1 = time.time()
            rep = factorial_replay_seed(env, seed, args.out, list(args.arms))
            print(json.dumps({"seed": seed, "replay_elapsed_s": time.time() - t1,
                              "all_deterministic": all(v["deterministic_replay"] for v in rep["cells"].values()),
                              "all_bit_exact": all(v["replay_states_bit_exact"] for v in rep["cells"].values())}), flush=True)
        env.close()
        return
    if args.mode == "train_worker":
        env = make_env(num_scenarios=scenario_count(seeds))
        stats = train_range(env, seeds, args.out, args.tag or f"{seeds[0]:07d}_{seeds[-1]:07d}")
        print(json.dumps(stats), flush=True)
        env.close()
        return
    # drivers
    log_dir = args.out / "logs"
    if args.mode == "factorial":
        per = args.seeds_per_proc or max(1, int(np.ceil(len(seeds) / args.procs)))
        groups = [seeds[i:i + per] for i in range(0, len(seeds), per)]
        cmds = []
        for g in groups:
            gen_cmd = [sys.executable, __file__, "factorial_worker", "--out", str(args.out), "--arms", args.arms, "--hazard-dist", str(args.hazard_dist),
                       "--hazard-body", args.hazard_body, *v09_argv, "--seeds", *map(str, g)] + (["--no-replay-check"] if args.no_replay_check else [])
            rep_cmd = [sys.executable, __file__, "factorial_replay", "--out", str(args.out), "--arms", args.arms, "--hazard-body", args.hazard_body,
                       *v09_argv, "--seeds", *map(str, g)]
            # generation and its fresh-process replay run sequentially (one job); jobs run --procs wide
            cmds.append(gen_cmd if args.no_replay_check else ["bash", "-c", shlex.join(map(str, gen_cmd)) + " && " + shlex.join(map(str, rep_cmd))])
        t0 = time.time()
        spawn_workers(cmds, args.procs, log_dir)
        prov = {**provenance(), "hazard_dist_m": args.hazard_dist, "seeds": seeds, "wall_s": time.time() - t0, "procs": args.procs}
        if V09_ACTIVE:
            prov["v09_flags"] = v09_flags
            prov["drawn_factors_by_seed"] = {str(s): draw_factors(s, RANDOMIZE) for s in seeds} if RANDOMIZE else None
        (args.out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
        print(json.dumps({"seeds": len(seeds), "wall_s": time.time() - t0, "seeds_per_s": len(seeds) / (time.time() - t0)}))
    elif args.mode == "train":
        groups = [seeds[i:i + args.clips_per_shard] for i in range(0, len(seeds), args.clips_per_shard)]
        cmds = [[sys.executable, __file__, "train_worker", "--out", str(args.out), "--hazard-body", args.hazard_body, "--steer-sigma", str(args.steer_sigma),
                 "--inlane-dist", str(args.inlane_dist[0]), str(args.inlane_dist[1]), "--p-throttle", str(args.p_throttle),
                 "--seeds", *map(str, g), "--tag", f"{g[0]:07d}_{g[-1]:07d}"] + (["--no-kind-pairing"] if args.no_kind_pairing else []) for g in groups]
        t0 = time.time()
        spawn_workers(cmds, args.procs, log_dir)
        stats = [json.loads(p.read_text()) for p in sorted(args.out.glob("stats_*.json"))]
        wall = time.time() - t0
        total_bytes = sum(w["bytes"] for s in stats for w in s["written"].values())
        n_clips_arm = sum(w["n"] for s in stats for k, w in s["written"].items() if k in ("armA", "shared"))
        summary = {"n_seeds": len(seeds), "n_clips_per_arm": n_clips_arm, "wall_s": wall, "clips_per_s_wall": n_clips_arm / wall,
                   "seeds_per_s_wall": len(seeds) / wall, "runs": sum(s["n_runs"] for s in stats),
                   "bytes": total_bytes, "mb_per_1k_clips_per_arm": total_bytes / max(1, n_clips_arm) * 1000 / 1e6, "procs": args.procs,
                   "distribution": TRAIN_DIST, "provenance": provenance()}
        (args.out / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k not in ("distribution", "provenance")}))


if __name__ == "__main__":
    main()
