#!/usr/bin/env python
"""MetaDrive feasibility spike for the pedestrian-hazard x ego-action factorial.

Checks (each a subcommand, `all` runs them in order and writes a JSON report):
  render      headless offscreen RGB (renderer backend, obs shape, fps @256px)
  ped         pedestrian spawn at lane-relative positions; pixel footprint vs distance
  factorial   {H1 in-lane, H0 sidewalk, H0' sidewalk-2} x {A0 brake, A1 throttle}
              -> speed profile, min distance, crash_human flag
  determinism two fresh processes, same seed + actions -> max |pixel diff|, state diff
  throughput  steps/s with / without rendering; 512->256 variant
  footprint   disk (env, assets), GPU memory used by the render context

Everything is CPU physics (Bullet); rendering goes through panda3d's EGL
(p3headlessgl) on the NVIDIA driver unless --tiny (p3tinydisplay, software).

Run on the Vast box with /opt/conda/envs/metadrive/bin/python.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------- config

CAM = 256


def make_env(seed=0, image=True, cam=CAM, lane_num=2, fov=None, tiny=False, outer_lane=True, map_region_size=1024):
    if tiny:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "load-display p3tinydisplay")
    from metadrive.envs import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    import logging
    cfg = dict(
        use_render=False,
        image_observation=image,
        norm_pixel=False,
        stack_size=1,
        traffic_density=0.0,
        accident_prob=0.0,
        num_scenarios=1,
        start_seed=seed,
        random_spawn_lane_index=False,
        random_lane_width=False,
        random_lane_num=False,
        random_agent_model=False,
        map_config=dict(type="block_sequence", config="SSS", lane_num=lane_num, lane_width=3.5),  # 3 straight blocks
        crash_human_done=False,
        crash_vehicle_done=False,
        crash_object_done=False,
        out_of_road_done=False,
        horizon=1000,
        show_interface=False,
        show_logo=False,
        show_fps=False,
        log_level=logging.WARNING,
        map_region_size=map_region_size,
    )
    if outer_lane:
        from metadrive.component.pgblock.first_block import FirstPGBlock
        cfg["agent_configs"] = {"default_agent": {"spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, lane_num - 1)}}
    if image:
        cfg["sensors"] = {"rgb_camera": (RGBCamera, cam, cam)}
        cfg["vehicle_config"] = {"image_source": "rgb_camera"}
    env = MetaDriveEnv(cfg)
    env.reset(seed=seed)
    if image and fov is not None:
        env.engine.get_sensor("rgb_camera").get_lens().setFov(fov)
    return env


def frame(obs):
    """uint8 HxWx3 from MetaDrive image obs (stack_size=1). MetaDrive returns BGR."""
    img = obs["image"]
    if img.ndim == 4:
        img = img[..., -1]
    return np.ascontiguousarray(img)


def save_png(path, img_bgr):
    import cv2
    cv2.imwrite(path, img_bgr)


def renderer_info(env):
    e = env.engine
    info = {}
    try:
        info["pipe"] = e.pipe.getType().getName()
        info["pipe_interface"] = e.pipe.getInterfaceName()
    except Exception as ex:  # pragma: no cover
        info["pipe_err"] = repr(ex)
    try:
        gsg = e.win.getGsg()
        info["driver_renderer"] = gsg.getDriverRenderer()
        info["driver_vendor"] = gsg.getDriverVendor()
        info["driver_version"] = gsg.getDriverVersion()
    except Exception as ex:  # pragma: no cover
        info["gsg_err"] = repr(ex)
    return info


def ego_state(env):
    a = env.agent
    v = a.velocity
    return np.array([a.position[0], a.position[1], a.heading_theta, v[0], v[1], float(a.speed)], dtype=np.float64)


def lane_frame(env):
    """Ego lane + outermost lane of the current road, ego longitudinal."""
    nav = env.agent.navigation
    ref = nav.current_ref_lanes
    ego_lane = env.agent.lane if env.agent.lane is not None else ref[0]
    long_, lat_ = ego_lane.local_coordinates(env.agent.position)
    return ego_lane, ref[-1], long_, lat_, ego_lane.width


def spawn_ped(env, kind, dist, heading=0.0, sidewalk_extra=1.0):
    """kind: H1 in ego lane `dist` m ahead; H0 sidewalk at matched longitudinal;
    H0p sidewalk, 1.5 m further out + heading flipped (second sidewalk pose)."""
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    ego_lane, outer, long_, lat_, w = lane_frame(env)
    if kind == "H1":
        pos = ego_lane.position(long_ + dist, 0.0)
    elif kind == "H0":
        pos = outer.position(long_ + dist, +(w / 2 + sidewalk_extra))
    elif kind == "H0p":
        pos = outer.position(long_ + dist, +(w / 2 + sidewalk_extra + 1.5))
        heading = heading + np.pi
    else:
        raise ValueError(kind)
    ped = env.engine.spawn_object(Pedestrian, position=pos, heading_theta=heading)
    return ped, np.asarray(pos, dtype=np.float64)


def matched_cone_class():
    """TrafficCone subclass with the pedestrian's collision envelope (cylinder r=0.35, h=1.75, 70 kg)
    and its visual rescaled to that envelope."""
    from metadrive.component.static_object.traffic_object import TrafficCone
    from metadrive.component.traffic_participants.pedestrian import Pedestrian

    class MatchedCone(TrafficCone):
        RADIUS = Pedestrian.RADIUS
        HEIGHT = Pedestrian.HEIGHT
        MASS = Pedestrian.MASS

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if self.render:
                # scale the visual (native cone model is ~0.7 m wide x 0.64 m tall at TrafficCone's scale,
                # NOT its nominal 2 m collision height) to exactly the pedestrian envelope
                for child in self.origin.getChildren():
                    if child.getName() != self.name:  # visual model instance
                        lo, hi = child.getTightBounds()
                        w, h = max(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]
                        sc = child.getScale()
                        child.setScale(sc[0] * 2 * self.RADIUS / w, sc[1] * 2 * self.RADIUS / w, sc[2] * self.HEIGHT / h)
                        child.setPos(0, 0, -self.HEIGHT / 2)
    return MatchedCone


def spawn_hazard(env, hazard, kind, dist, heading=0.0, solid=True, kinematic=False, static=True):
    """hazard in {ped, cone}; kind in {H1, H0, H0p}. solid=False -> intangible: collide mask cleared
    (Bullet groups-mask filter => no pair with anything, incl. vehicle contactTest) and kinematic so
    gravity does not drop it through the terrain. kinematic=True also for solid so both variants
    share the exact same (externally driven) motion => identical renders up to contact."""
    from panda3d.core import BitMask32
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    cls = Pedestrian if hazard == "ped" else matched_cone_class()
    ego_lane, outer, long_, lat_, w = lane_frame(env)
    if kind == "H1":
        pos = ego_lane.position(long_ + dist, 0.0)
    elif kind == "H0":
        pos = outer.position(long_ + dist, +(w / 2 + 1.0))  # sidewalk (2 m wide) starts at +w/2 of the outer lane
    elif kind == "H0p":
        pos = outer.position(long_ + dist, +(w / 2 + 2.5))
        heading = heading + np.pi
    else:
        raise ValueError(kind)
    obj = env.engine.spawn_object(cls, position=pos, heading_theta=heading)
    world = env.engine.physics_world.dynamic_world
    if kinematic:  # NOTE: setKinematic after attach does NOT collide unless the body is re-attached
        world.remove(obj.body); obj.body.setKinematic(True); world.attach(obj.body)
    elif static:
        obj.body.setStatic(True)
    if not solid:
        # ghost collision group: pairs only with Terrain (so it rests on the ground), never with the vehicle,
        # so contactTest / crash flags / physics response are all absent; render is untouched
        from metadrive.constants import CollisionGroup
        g = 12
        obj.body.setIntoCollideMask(BitMask32.bit(g))
        for other in range(32):
            world.setGroupCollisionFlag(g, other, other == CollisionGroup.Terrain.getLowestOnBit())
    return obj, np.asarray(pos, dtype=np.float64)


def set_ego_speed(env, v_mps):
    env.agent.set_velocity([1.0, 0.0], v_mps, in_local_frame=True)


def gpu_mem_for_pid(pid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        out2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except Exception as ex:
        return {"err": repr(ex)}
    mine = [l for l in out2.splitlines() if f" {pid} " in l]
    return {"compute_apps": out.strip(), "nvidia_smi_rows_for_pid": mine}


# ----------------------------------------------------------------------------- checks


def check_render(out, cam=CAM, tiny=False, nsteps=100):
    t0 = time.time()
    env = make_env(image=True, cam=cam, tiny=tiny)
    t_make = time.time() - t0
    obs, _ = env.reset(seed=0)
    img = frame(obs)
    res = {"cam": cam, "tiny": tiny, "env_create_s": t_make, "obs_shape": list(obs["image"].shape),
           "obs_dtype": str(obs["image"].dtype), "renderer": renderer_info(env),
           "pid": os.getpid(), "gpu": gpu_mem_for_pid(os.getpid()),
           "frame_mean": float(img.mean()), "frame_std": float(img.std())}
    save_png(os.path.join(out, f"render_{cam}{'_tiny' if tiny else ''}.png"), img)
    t0 = time.time()
    for _ in range(nsteps):
        obs, r, term, trunc, info = env.step([0.0, 0.3])
    dt = time.time() - t0
    res["steps"] = nsteps
    res["steps_per_s_render"] = nsteps / dt
    res["frame_after_mean"] = float(frame(obs).mean())
    env.close()
    return res


def check_ped(out, dists=(4, 6, 8, 10, 12, 15, 20, 25, 30), fovs=(None, 40.0), cam=CAM):
    """Footprint by rendered difference: same seed/action with vs without the pedestrian."""
    res = {"cam": cam, "rows": []}
    for fov in fovs:
        env = make_env(image=True, cam=cam, fov=fov)
        lens = env.engine.get_sensor("rgb_camera").get_lens()
        fov_eff = float(lens.getFov()[0])
        # baseline frame (no ped) after one zero step
        env.reset(seed=0)
        obs, *_ = env.step([0.0, 0.0])
        base = frame(obs).astype(np.int16)
        for kind in ("H1", "H0", "H0p"):
            for d in dists:
                env.reset(seed=0)
                ped, pos = spawn_ped(env, kind, d)
                obs, *_ = env.step([0.0, 0.0])
                img = frame(obs)
                diff = np.abs(img.astype(np.int16) - base).max(-1)
                mask = diff > 8
                n = int(mask.sum())
                row = {"fov": fov_eff, "kind": kind, "dist_m": d, "pixels": n,
                       "ped_pos": pos.tolist(), "ego_pos": [float(x) for x in env.agent.position],
                       "center_dist_m": float(np.linalg.norm(pos - np.asarray(env.agent.position)))}
                if n:
                    ys, xs = np.where(mask)
                    row.update({"bbox_w": int(xs.max() - xs.min() + 1), "bbox_h": int(ys.max() - ys.min() + 1),
                                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
                res["rows"].append(row)
                if kind == "H1" or d in (10, 20):
                    save_png(os.path.join(out, f"ped_{kind}_d{d}_fov{int(fov_eff)}.png"), img)
                env.engine.clear_objects([ped.id])
        env.close()
    return res


def run_cell(env, kind, dist, action_kind, ctx_steps=15, chunk_steps=35, ctx_throttle=0.8, throttle=0.8,
             record_frames=False):
    env.reset(seed=0)
    ped, ppos = spawn_ped(env, kind, dist) if kind != "none" else (None, None)
    speeds, dists, flags, frames, states = [], [], [], [], []
    ego_len = env.agent.LENGTH
    for t in range(ctx_steps + chunk_steps):
        if t < ctx_steps:
            act = [0.0, ctx_throttle]
        else:
            act = [0.0, throttle] if action_kind == "A1" else [0.0, -1.0]
        obs, r, term, trunc, info = env.step(act)
        st = ego_state(env)
        states.append(st)
        speeds.append(float(env.agent.speed))
        if ped is not None:
            d = float(np.linalg.norm(np.asarray(env.agent.position) - np.asarray(ped.position)))
            dists.append(d - ego_len / 2 - ped.RADIUS)
        flags.append({"crash_human": bool(env.agent.crash_human), "info_crash_human": bool(info.get("crash_human", False)),
                      "crash_sidewalk": bool(env.agent.crash_sidewalk), "out_of_road": bool(info.get("out_of_road", False))})
        if record_frames:
            frames.append(frame(obs).copy())
    out = {"kind": kind, "action": action_kind, "dist_m": dist, "ctx_steps": ctx_steps, "chunk_steps": chunk_steps,
           "speed_kmh": speeds, "min_gap_m": (float(min(dists)) if dists else None),
           "argmin_gap_step": (int(np.argmin(dists)) if dists else None),
           "crash_human_any": any(f["crash_human"] for f in flags),
           "info_crash_human_any": any(f["info_crash_human"] for f in flags),
           "first_crash_step": next((i for i, f in enumerate(flags) if f["crash_human"]), None),
           "final_speed_kmh": speeds[-1], "peak_speed_kmh": max(speeds)}
    if ped is not None:
        env.engine.clear_objects([ped.id])
    return out, np.stack(states), (np.stack(frames) if frames else None)


def check_factorial(out, dist=25.0, cam=CAM):
    env = make_env(image=True, cam=cam)
    res = {"dist_m": dist, "cells": [], "flags_api": "env.agent.crash_human (BaseVehicle attr, set in _body_contact on "
                                                    "MetaDriveType.PEDESTRIAN/CYCLIST contact); info['crash_human'] "
                                                    "from done_function; crash_human_done=False keeps stepping"}
    ctx_frames = {}
    for kind in ("H1", "H0", "H0p"):
        for act in ("A0", "A1"):
            cell, states, frames = run_cell(env, kind, dist, act, record_frames=True)
            res["cells"].append(cell)
            ctx_frames[(kind, act)] = frames[:15]
            save_png(os.path.join(out, f"cell_{kind}_{act}_t{len(frames)-1}.png"), frames[-1])
            save_png(os.path.join(out, f"cell_{kind}_{act}_t15.png"), frames[15])
    # identical pre-hazard context across the 6 cells? (pedestrian is present from t=0 here, so
    # H1 vs H0 context frames legitimately differ if the ped is visible; A0 vs A1 must be identical)
    keys = list(ctx_frames)
    ctx_diff = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ctx_diff[f"{keys[i]}|{keys[j]}"] = int(np.abs(ctx_frames[keys[i]].astype(np.int16) - ctx_frames[keys[j]].astype(np.int16)).max())
    res["context_frame_max_absdiff"] = ctx_diff
    env.close()
    return res


def check_hazard_spawn_midepisode(out, dist=20.0, cam=CAM):
    """Branch-at-T: identical prefix (no ped), spawn ped at t=T in one branch only -> prefix frames identical."""
    env = make_env(image=True, cam=cam)
    T = 12
    runs = {}
    for branch in ("noped", "ped_at_T"):
        env.reset(seed=0)
        frames, states = [], []
        ped = None
        for t in range(T + 20):
            if t == T and branch == "ped_at_T":
                ped, _ = spawn_ped(env, "H1", dist)
            obs, *_ = env.step([0.0, 0.8])
            frames.append(frame(obs).copy())
            states.append(ego_state(env))
        runs[branch] = (np.stack(frames), np.stack(states))
        if ped is not None:
            env.engine.clear_objects([ped.id])
    fa, sa = runs["noped"]
    fb, sb = runs["ped_at_T"]
    pre = int(np.abs(fa[:T].astype(np.int16) - fb[:T].astype(np.int16)).max())
    post = int(np.abs(fa[T:].astype(np.int16) - fb[T:].astype(np.int16)).max())
    env.close()
    return {"T": T, "prefix_frames_max_absdiff": pre, "prefix_state_max_absdiff": float(np.abs(sa[:T] - sb[:T]).max()),
            "post_spawn_frames_max_absdiff": post,
            "note": "MetaDrive has no Bullet save/restore; branching = replay identical prefix (deterministic) then "
                    "spawn_object at T. engine.record_episode/replay exists but replays trajectories, not physics forks."}


def check_throughput(out, cam=CAM, nsteps=300):
    res = {}
    for label, image, c in (("no_render", False, cam), ("render_256", True, 256), ("render_512", True, 512)):
        env = make_env(image=image, cam=c)
        env.reset(seed=0)
        env.step([0.0, 0.3])
        t0 = time.time()
        for _ in range(nsteps):
            obs, *_ = env.step([0.0, 0.3])
            if image and c == 512:
                import cv2
                cv2.resize(frame(obs), (256, 256), interpolation=cv2.INTER_AREA)
        dt = time.time() - t0
        res[label] = {"steps_per_s": nsteps / dt, "reset_s": None}
        t0 = time.time()
        env.reset(seed=0)
        res[label]["reset_s"] = time.time() - t0
        env.close()
    sps = res["render_256"]["steps_per_s"]
    res["estimate_10k_clips_x_20_frames"] = {"steps": 200_000, "wall_h_single_proc": 200_000 / sps / 3600,
                                             "plus_resets_h_single_proc": 10_000 * res["render_256"]["reset_s"] / 3600}
    return res


def check_footprint(out):
    import metadrive
    md = os.path.dirname(metadrive.__file__)
    env_dir = os.path.abspath(os.path.join(md, "..", "..", "..", ".."))

    def du(p):
        return subprocess.run(["du", "-sh", p], capture_output=True, text=True).stdout.split()[0]

    return {"metadrive_pkg": du(md), "assets": du(os.path.join(md, "assets")), "conda_env": du(env_dir),
            "assets_url": "https://github.com/metadriverse/metadrive/releases/download/MetaDrive-0.4.3/assets.zip"}


def run_toggle(env, hazard, kind, action_kind, solid, dist=20.0, v0=8.0, ctx_steps=10, chunk_steps=40,
               walking=False, walk_speed=1.2):
    """Ego starts at v0 m/s; context = hold throttle; chunk = A1 throttle / A0 full brake."""
    env.reset(seed=0)
    set_ego_speed(env, v0)
    obj = None
    if hazard != "none":
        # walking: start on the sidewalk side, cross toward/through the ego lane
        obj, p0 = spawn_hazard(env, hazard, "H0" if walking else kind, dist, solid=solid,
                               heading=(np.pi / 2 if walking else 0.0))
        if walking:
            ego_lane, outer, long_, lat_, w = lane_frame(env)
            p_end = ego_lane.position(long_ + dist, 0.0)
            walk_dir = (p_end - p0) / np.linalg.norm(p_end - p0)
            if hazard == "ped":
                obj.set_velocity(walk_dir.tolist(), walk_speed)  # switches to the walking animation model
                obj.body.setKinematic(True)
    frames, states, flags, cdist = [], [], [], []
    dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
    for t in range(ctx_steps + chunk_steps):
        act = [0.0, 0.5] if (t < ctx_steps or action_kind == "A1") else [0.0, -1.0]
        if obj is not None and walking:
            obj.set_position(p0 + walk_dir * walk_speed * dt * (t + 1), height=obj.HEIGHT / 2)
        obs, r, term, trunc, info = env.step(act)
        frames.append(frame(obs).copy())
        states.append(ego_state(env))
        a = env.agent
        flags.append((bool(a.crash_human), bool(a.crash_object), bool(a.crash_sidewalk)))
        if obj is not None:
            cdist.append(float(np.linalg.norm(np.asarray(a.position) - np.asarray(obj.position))))
    if obj is not None:
        env.engine.clear_objects([obj.id])
    first = next((i for i, f in enumerate(flags) if f[0] or f[1]), None)
    return {"hazard": hazard, "kind": kind, "action": action_kind, "solid": solid, "walking": walking,
            "crash_human": any(f[0] for f in flags), "crash_object": any(f[1] for f in flags),
            "first_contact_step": first, "min_center_dist_m": (min(cdist) if cdist else None),
            "speed_kmh_at": {str(i): round(float(states[i][5]), 2) for i in (0, 9, 19, 29, 39, 49)},
            "hazard_pos": (obj is not None and p0.tolist())}, np.stack(frames), np.stack(states)


def check_toggle(out, cam=CAM, dist=20.0):
    env = make_env(image=True, cam=cam, fov=40.0)
    runs = {}
    base = {}
    for act in ("A0", "A1"):
        base[act] = run_toggle(env, "none", "H1", act, True, dist=dist)
    res = {"dist_m": dist, "v0_mps": 8.0, "fov": 40, "cells": [], "pairs": [], "baseline": {a: base[a][0] for a in base}}
    specs = [(h, k, a, w) for h in ("ped", "cone") for k in ("H1", "H0") for a in ("A0", "A1") for w in (False,)]
    specs += [("ped", "H1", "A1", True), ("ped", "H1", "A0", True)]
    for h, k, a, w in specs:
        pair = {}
        for solid in (True, False):
            r, fr, st = run_toggle(env, h, k, a, solid, dist=dist, walking=w)
            res["cells"].append(r)
            pair[solid] = (r, fr, st)
            if solid and a == "A1":
                save_png(os.path.join(out, f"toggle_{h}_{k}_{a}{'_walk' if w else ''}_t{len(fr)-1}.png"), fr[-1])
        rs, fs, ss = pair[True]
        ri, fi, si = pair[False]
        T = rs["first_contact_step"]
        upto = T if T is not None else len(fs)
        bf, bs = base[a][1], base[a][2]
        res["pairs"].append({
            "hazard": h, "kind": k, "action": a, "walking": w, "solid_first_contact_step": T,
            "solid_crash_human": rs["crash_human"], "solid_crash_object": rs["crash_object"],
            "intangible_crash_any": ri["crash_human"] or ri["crash_object"],
            "intangible_min_center_dist_m": ri["min_center_dist_m"], "solid_min_center_dist_m": rs["min_center_dist_m"],
            "frames_max_absdiff_before_contact": int(np.abs(fs[:upto].astype(np.int16) - fi[:upto].astype(np.int16)).max()) if upto else 0,
            "frames_frac_px_diff_before_contact": float((np.abs(fs[:upto].astype(np.int16) - fi[:upto].astype(np.int16)).max(-1) > 0).mean()) if upto else 0.0,
            "ego_state_max_absdiff_before_contact": float(np.abs(ss[:upto] - si[:upto]).max()) if upto else 0.0,
            "ego_state_identical_full_traj_solid_vs_intangible": bool(np.array_equal(ss, si)),
            "intangible_vs_nohazard_state_max_absdiff": float(np.abs(si - bs).max()),
            "intangible_vs_nohazard_frames_max_absdiff": int(np.abs(fi.astype(np.int16) - bf.astype(np.int16)).max()),
            "solid_speed_kmh_after_contact": (rs["speed_kmh_at"]),
        })
    # hazard visibility at the context end (t=ctx-1) for ped vs cone (H1, fov 40, dist 20 at v0=8 -> ~12 m at t=9)
    for h in ("ped", "cone"):
        r, fr, st = run_toggle(env, h, "H1", "A0", True, dist=dist)
        d = np.abs(fr[9].astype(np.int16) - base["A0"][1][9].astype(np.int16)).max(-1) > 8
        ys, xs = np.where(d)
        res[f"{h}_footprint_t9"] = {"pixels": int(d.sum()), "bbox_w": int(xs.max() - xs.min() + 1) if len(xs) else 0,
                                    "bbox_h": int(ys.max() - ys.min() + 1) if len(ys) else 0,
                                    "center_dist_m": float(np.linalg.norm(st[9][:2] - np.asarray(r["hazard_pos"])))}
        save_png(os.path.join(out, f"vis_{h}_t9.png"), fr[9])
    env.close()
    return res


def _run_hazard_variant(env, hazard, variant, dist=20.0, v0=8.0, steps=50, walking=False, pose_anim=False):
    """variant: dynamic | kinematic | kinematic_reattach | static | ghost_mask_off | ghost_group | none"""
    from panda3d.core import BitMask32
    from metadrive.constants import CollisionGroup
    env.reset(seed=0)
    set_ego_speed(env, v0)
    obj = None
    z0 = None
    if variant != "none":
        obj, p0 = spawn_hazard(env, hazard, "H0" if walking else "H1", dist, solid=True, kinematic=False,
                               heading=(np.pi / 2 if walking else 0.0))
        body = obj.body
        world = env.engine.physics_world.dynamic_world
        if variant == "kinematic":
            body.setKinematic(True)
        elif variant == "kinematic_reattach":
            world.remove(body); body.setKinematic(True); world.attach(body)
        elif variant == "static":
            body.setStatic(True)
        elif variant == "ghost_mask_off":
            body.setIntoCollideMask(BitMask32.allOff())
        elif variant == "ghost_group":
            g = 12
            body.setIntoCollideMask(BitMask32.bit(g))
            for other in range(32):
                world.setGroupCollisionFlag(g, other, other == CollisionGroup.Terrain.getLowestOnBit())
        if walking:
            ego_lane, outer, long_, lat_, w = lane_frame(env)
            p_end = ego_lane.position(long_ + dist, 0.0)
            wd = (p_end - p0) / np.linalg.norm(p_end - p0)
            if hazard == "ped":
                obj.set_velocity(wd.tolist(), 1.2)
                if pose_anim:
                    from metadrive.component.traffic_participants.pedestrian import Pedestrian
                    ac = Pedestrian._MODEL[obj.current_speed_model].get_anim_control("Take 001")
                    ac.stop()
        z0 = float(obj.origin.getZ())
    frames, states, flags, cdist, zs = [], [], [], [], []
    for t in range(steps):
        if obj is not None and walking:
            if hazard == "ped":
                obj.set_velocity(wd.tolist(), 1.2)
                if pose_anim:
                    ac.pose(t % max(1, ac.getNumFrames()))
        obs, r, term, trunc, info = env.step([0.0, 0.5])
        frames.append(frame(obs).copy()); states.append(ego_state(env))
        a = env.agent
        flags.append((bool(a.crash_human), bool(a.crash_object)))
        if obj is not None:
            cdist.append(float(np.linalg.norm(np.asarray(a.position) - np.asarray(obj.position))))
            zs.append(float(obj.origin.getZ()))
    if obj is not None:
        env.engine.clear_objects([obj.id])
    first = next((i for i, f in enumerate(flags) if f[0] or f[1]), None)
    return {"hazard": hazard, "variant": variant, "walking": walking, "pose_anim": pose_anim,
            "crash_human": any(f[0] for f in flags), "crash_object": any(f[1] for f in flags),
            "first_contact_step": first, "min_center_dist_m": (min(cdist) if cdist else None),
            "hazard_z_start_end": ([z0, zs[-1]] if zs else None),
            "ego_speed_kmh_t19_t29_t49": [round(float(states[i][5]), 2) for i in (19, 29, 49)]}, np.stack(frames), np.stack(states)


def check_contact(out, cam=CAM, dist=20.0):
    res = {"variants": [], "notes": []}
    env = make_env(image=True, cam=cam, fov=40.0)
    base = {}
    for h in ("ped", "cone"):
        base[h] = _run_hazard_variant(env, h, "none", dist)
        for v in ("dynamic", "kinematic_reattach", "static", "ghost_mask_off", "ghost_group"):
            r, fr, st = _run_hazard_variant(env, h, v, dist)
            r["ego_state_maxdiff_vs_nohazard"] = float(np.abs(st - base[h][2]).max())
            res["variants"].append(r)
            if v in ("dynamic", "static"):
                save_png(os.path.join(out, f"contact_{h}_{v}_t{len(fr)-1}.png"), fr[-1])
    # solid(dynamic or static) vs ghost_group: frames identical before contact?
    for h in ("ped", "cone"):
        for solid_v in ("dynamic", "static"):
            rs, fs, ss = _run_hazard_variant(env, h, solid_v, dist)
            ri, fi, si = _run_hazard_variant(env, h, "ghost_group", dist)
            T = rs["first_contact_step"]
            upto = T if T is not None else len(fs)
            res["variants"].append({"pair": f"{h}:{solid_v}-vs-ghost_group", "solid_first_contact_step": T,
                                    "frames_max_absdiff_before_contact": int(np.abs(fs[:upto].astype(np.int16) - fi[:upto].astype(np.int16)).max()),
                                    "n_px_diff_before_contact": int((np.abs(fs[:upto].astype(np.int16) - fi[:upto].astype(np.int16)).max(-1) > 0).sum()),
                                    "ego_state_max_absdiff_before_contact": float(np.abs(ss[:upto] - si[:upto]).max()),
                                    "ego_speed_solid_after": rs["ego_speed_kmh_t19_t29_t49"], "ego_speed_ghost_after": ri["ego_speed_kmh_t19_t29_t49"]})
    # walking pedestrian: animation determinism (two identical runs) with and without explicit posing
    for pose in (False, True):
        r1, f1, s1 = _run_hazard_variant(env, "ped", "ghost_group", dist, walking=True, pose_anim=pose)
        r2, f2, s2 = _run_hazard_variant(env, "ped", "ghost_group", dist, walking=True, pose_anim=pose)
        res["variants"].append({"walking_repeat_pose_anim": pose, "frames_max_absdiff": int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max()),
                                "n_px_diff": int((np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max(-1) > 0).sum()),
                                "state_identical": bool(np.array_equal(s1, s2)), "min_center_dist": r1["min_center_dist_m"], "z": r1["hazard_z_start_end"]})
        save_png(os.path.join(out, f"walk_pose{int(pose)}_t30.png"), f1[30])
    # cone visual: children of origin
    env.reset(seed=0)
    cone, _ = spawn_hazard(env, "cone", "H1", 12.0, solid=True, kinematic=False)
    kids = [(c.getName(), [round(x, 3) for x in c.getScale()], [round(x, 2) for x in c.getPos()]) for c in cone.origin.getChildren()]
    res["cone_origin_children"] = kids
    b0 = cone.origin.getTightBounds()
    res["cone_tight_bounds"] = [[round(x, 2) for x in b0[0]], [round(x, 2) for x in b0[1]]] if b0 else None
    env.engine.clear_objects([cone.id])
    env.close()
    # reset cost vs map_region_size
    for mrs in (1024, 512, 256):
        from metadrive.envs import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
        import logging
        try:
            e = MetaDriveEnv(dict(use_render=False, image_observation=True, norm_pixel=False, stack_size=1, traffic_density=0.0,
                                  num_scenarios=1, start_seed=0, map_config=dict(type="block_sequence", config="SSS", lane_num=2, lane_width=3.5),
                                  sensors={"rgb_camera": (RGBCamera, cam, cam)}, vehicle_config={"image_source": "rgb_camera"},
                                  map_region_size=mrs, log_level=logging.WARNING, show_interface=False, show_logo=False, show_fps=False))
            e.reset(seed=0)
            t0 = time.time(); e.reset(seed=0); e.reset(seed=0); dt = (time.time() - t0) / 2
            res["notes"].append({"map_region_size": mrs, "reset_s": dt})
            e.close()
        except Exception as ex:
            res["notes"].append({"map_region_size": mrs, "error": repr(ex)[:300]})
    return res


def check_softreset(out, cam=CAM, n=20):
    """Avoid the 4.5 s render-mode env.reset(): reposition the ego instead. Verify state parity."""
    env = make_env(image=True, cam=cam)
    obs0, _ = env.reset(seed=0)
    s_hard = ego_state(env)
    f_hard = frame(obs0).copy()
    for _ in range(10):
        env.step([0.3, 0.8])
    a = env.agent
    lane = a.navigation.current_ref_lanes[0]
    t0 = time.time()
    for _ in range(n):
        a.set_position(lane.position(*lane.local_coordinates(np.asarray(s_hard[:2]))), None)
        a.set_heading_theta(s_hard[2])
        a.set_velocity([0.0, 0.0], 0.0)
        a.set_angular_velocity(0.0)
        a.set_pitch(0.0) if hasattr(a, "set_pitch") else None
        a.set_roll(0.0) if hasattr(a, "set_roll") else None
    dt = (time.time() - t0) / n
    obs, *_ = env.step([0.0, 0.0])
    s_soft = ego_state(env)
    env.reset(seed=0)
    obs_h, *_ = env.step([0.0, 0.0])
    s_hard1 = ego_state(env)
    env.close()
    return {"soft_reset_s": dt, "state_after_soft_minus_hard_step": np.abs(s_soft - s_hard1).tolist(),
            "frame_max_absdiff_soft_vs_hard": int(np.abs(frame(obs).astype(np.int16) - frame(obs_h).astype(np.int16)).max())}


def determinism_worker(path, cam=CAM, tiny=False):
    env = make_env(image=True, cam=cam, tiny=tiny, fov=40.0)
    cell, frames, states = run_toggle(env, "ped", "H1", "A1", True, dist=20.0)
    np.savez_compressed(path, states=states, frames=frames)
    env.close()


def check_determinism(out, cam=CAM, tiny=False):
    paths = [os.path.join(out, f"det_run{i}.npz") for i in range(2)]
    for p in paths:
        cmd = [sys.executable, __file__, "det_worker", "--out", out, "--path", p, "--cam", str(cam)] + (["--tiny"] if tiny else [])
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError("det_worker failed: " + cp.stderr[-3000:])
    a, b = [np.load(p) for p in paths]
    fd = np.abs(a["frames"].astype(np.int16) - b["frames"].astype(np.int16))
    sd = np.abs(a["states"] - b["states"])
    return {"frames": list(a["frames"].shape), "max_abs_pixel_diff": int(fd.max()),
            "frac_pixels_differing": float((fd.max(-1) > 0).mean()), "max_abs_state_diff": float(sd.max()),
            "states_bit_identical": bool(np.array_equal(a["states"], b["states"])),
            "frames_bit_identical": bool(np.array_equal(a["frames"], b["frames"]))}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "render", "ped", "factorial", "branch", "toggle", "contact", "softreset", "determinism",
                                    "det_worker", "throughput", "footprint"])
    ap.add_argument("--out", default="/root/cgs-pilot/artifacts/metadrive_spike")
    ap.add_argument("--path")
    ap.add_argument("--cam", type=int, default=CAM)
    ap.add_argument("--tiny", action="store_true", help="force p3tinydisplay software renderer")
    ap.add_argument("--dist", type=float, default=25.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "det_worker":
        determinism_worker(args.path, cam=args.cam, tiny=args.tiny)
        return
    report = {}
    steps = {
        "render": lambda: check_render(args.out, cam=args.cam, tiny=args.tiny),
        "ped": lambda: check_ped(args.out, cam=args.cam),
        "factorial": lambda: check_factorial(args.out, dist=args.dist, cam=args.cam),
        "branch": lambda: check_hazard_spawn_midepisode(args.out, cam=args.cam),
        "toggle": lambda: check_toggle(args.out, cam=args.cam),
        "contact": lambda: check_contact(args.out, cam=args.cam),
        "softreset": lambda: check_softreset(args.out, cam=args.cam),
        "determinism": lambda: check_determinism(args.out, cam=args.cam, tiny=args.tiny),
        "throughput": lambda: check_throughput(args.out, cam=args.cam),
        "footprint": lambda: check_footprint(args.out),
    }
    todo = list(steps) if args.cmd == "all" else [args.cmd]
    for name in todo:
        t0 = time.time()
        try:
            report[name] = steps[name]()
        except Exception as ex:
            import traceback
            report[name] = {"error": repr(ex), "traceback": traceback.format_exc()}
        report[name]["_elapsed_s"] = time.time() - t0
        print(f"=== {name} ({report[name]['_elapsed_s']:.1f}s)")
        print(json.dumps(report[name], indent=1, default=str)[:6000])
        sys.stdout.flush()
    tag = "all" if args.cmd == "all" else args.cmd
    with open(os.path.join(args.out, f"report_{tag}{'_tiny' if args.tiny else ''}.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
