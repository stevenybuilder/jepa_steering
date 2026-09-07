#!/usr/bin/env python
"""MetaDrive-side bridge for ``closed_loop_rollout.py`` (runs in the MetaDrive env; no torch).

The model env (``/opt/conda/bin/python``) and the MetaDrive env (``/opt/conda/envs/metadrive/bin/python``) do not share
torch, so the closed loop is bridged by this small JSON-lines server: one request line on stdin -> one response line on
stdout; everything else (MetaDrive / panda3d chatter) goes to stderr (fd 1 is dup'ed to fd 2 at start-up).

Every episode replays the generator's prefix EXACTLY as ``metadrive_hazard_pilot.run_episode`` does (reset_scene ->
speed seeding -> hazard spawn at ``s_ahead + prefix_travel`` -> pedestrian pose -> PREFIX_STEPS steps of
``[0, prefix_throttle]``), because MetaDrive has no Bullet save/restore: "branching from the context state" = deterministic
prefix replay (DRIVING_METADRIVE_SPIKE.md s4). The context state is returned so the driver can check it against the stored
``ego_states`` before trusting the rollout.

Protocol (all floats are JSON numbers; frames / masks are base64 of raw bytes / packbits):
  {"cmd": "init", "closed_loop": true}
      -> {"ok": true, "provenance": {...}, "termination": {...}, "horizon_guard": 1000}
      closed_loop = MetaDrive's DEFAULT termination flags (crash_human_done / crash_object_done / crash_vehicle_done /
      out_of_road_done = True; the generator's make_env sets them False only for its scripted futures) so every episode ends
      at the env's own ``done``; the generator's horizon (1000 env steps) is kept as an engineering guard against an infinite
      stall (MetaDrive's shipped default is None) and surfaces as the NATIVE ``max_step`` predicate.
  {"cmd": "reset", "seed": int, "map_cfg": str, "row": {manifest row or factor dict}, "v0": float,
   "hazard": {"kind": "ped"|"cone", "s_ahead": m, "lateral": m, "solid": bool, "prefix_travel": m} | null}
      -> {"states": [[x,y,z,heading,vx,vy,speed] x (PREFIX_STEPS+1)], "crash_human": [...], "crash_object": [...],
          "crash_sidewalk": [...], "out_of_road": [...], "gap"/"center_dist"/"lon"/"lat": [...] (hazard only),
          "hazard_pose": [x,y,z,h] | null, "hazard_px": [u,v,ok] | null, "frame": b64 uint8[256,256,3] RGB,
          "hazard_mask": b64 packbits[256,256], "corridor_mask": b64 packbits[256,256], "ctx_long": m, "ctx_lat": m,
          "prefix_travel_measured": m, "lane_length": m, "ego_dims": [L, W]}
  {"cmd": "step", "chunk": [[steer, throttle_brake], ...], "want_frames": true}   (one entry per MODEL step; 3 sim steps each)
      -> {"steps": [{"state": [...7], "crash_human", "crash_object", "crash_sidewalk", "out_of_road", "gap", "center_dist",
                     "lon", "lat", "progress_m", "disp_m", "speed", "terminated", "truncated", "done", "done_info" (the
                     TerminationState predicates from info), "route_completion", "env_episode_step"} per sim step],
          "done": bool (stepping stops at the env's own done), "frames": [b64 per completed model step], "hazard_mask", ...}
  {"cmd": "close"} -> {"ok": true}

``progress_m`` = longitudinal coordinate along the CONTEXT lane (``lane.local_coordinates``) minus the context value; the
first blocks are straight and collinear (metadrive_hazard_pilot.lane_pose), so this is exact for the < 70 m we travel.
``disp_m`` = Euclidean displacement from the context position (reported alongside).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# protocol lines go to the ORIGINAL stdout; fd 1 is redirected to stderr so no library output can corrupt the stream
_PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import metadrive_hazard_pilot as gen  # noqa: E402  (after the fd juggling; imports panda3d lazily)


TERMINATION_FLAGS = ("crash_human_done", "crash_object_done", "crash_vehicle_done", "out_of_road_done")
TERMINATION_KEYS = ("arrive_dest", "out_of_road", "max_step", "crash", "crash_vehicle", "crash_human", "crash_object", "crash_building", "crash_sidewalk")


def b64_array(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def b64_mask(m: np.ndarray) -> str:
    return base64.b64encode(np.packbits(np.asarray(m, dtype=bool).reshape(-1)).tobytes()).decode("ascii")


def log(msg: str) -> None:
    print(f"[bridge {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Session:
    def __init__(self) -> None:
        self.env = None
        self.obj = None
        self.radius = 0.0
        self.ctx_lane = None
        self.ctx_long = 0.0
        self.ctx_pos = None
        self.hazard = None
        self.closed_loop = False
        self.steps_after_context = 0

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _clear_hazard(self) -> None:
        if self.obj is not None and self.env is not None:
            # run_episode clears with force_destroy: pooled reuse re-applies setStatic/ghost mask -> divergent physics
            self.env.engine.clear_objects([self.obj.id], force_destroy=True)
            self.obj = None

    def _capture(self, obs) -> dict[str, Any]:
        env = self.env
        out = {"frame": b64_array(gen.frame_rgb(obs)), "corridor_mask": b64_mask(gen.corridor_mask(env))}
        sem = gen.semantic_rgb(env)
        if self.obj is not None:
            colour = gen.semantic_colour("PEDESTRIAN" if self.hazard["kind"] == "ped" else gen.CONE_SEMANTIC)
            out["hazard_mask"] = b64_mask(gen.colour_mask(sem, colour))
            u, v, ok = gen.project(env, (self.obj.position[0], self.obj.position[1], self.obj.origin.getZ()))
            out["hazard_px"] = [float(u), float(v), bool(ok)]
        else:
            out["hazard_mask"] = b64_mask(np.zeros((gen.IMAGE_SIZE, gen.IMAGE_SIZE), dtype=bool))
            out["hazard_px"] = None
        return out

    def _step_record(self, info, terminated: bool = False, truncated: bool = False) -> dict[str, Any]:
        env = self.env
        a = env.agent
        st = gen.ego_state(env)
        rec: dict[str, Any] = {
            "state": st.tolist(), "speed": float(st[6]),
            "crash_human": bool(a.crash_human), "crash_object": bool(a.crash_object), "crash_sidewalk": bool(a.crash_sidewalk),
            "out_of_road": bool(info.get("out_of_road", False)),
            "terminated": bool(terminated), "truncated": bool(truncated), "done": bool(terminated or truncated),
            "done_info": {k: bool(info.get(k, False)) for k in TERMINATION_KEYS},
            "route_completion": (float(info["route_completion"]) if info.get("route_completion") is not None else None),
            "env_episode_step": int(getattr(env, "episode_step", -1)),
        }
        if self.obj is not None:
            dc, lon, lat, gap = gen.hazard_gap(env, self.obj, self.radius)
            rec.update({"center_dist": dc, "lon": lon, "lat": lat, "gap": gap, "hazard_z": float(self.obj.origin.getZ())})
        if self.ctx_lane is not None:
            long_, lat_ = self.ctx_lane.local_coordinates(a.position)
            rec["progress_m"] = float(long_ - self.ctx_long)
            rec["lane_lat_m"] = float(lat_)
            rec["disp_m"] = float(np.hypot(*(np.asarray(a.position, dtype=np.float64) - self.ctx_pos)))
        return rec

    # ---- commands -------------------------------------------------------------------------------------------------
    def init(self, req: dict[str, Any]) -> dict[str, Any]:
        t0 = time.time()
        self.env = gen.make_env(num_scenarios=int(req.get("num_scenarios", 200_000)))
        self.closed_loop = bool(req.get("closed_loop", False))
        if self.closed_loop:  # MetaDrive's default termination semantics (the generator disables them for scripted futures)
            for k in TERMINATION_FLAGS:
                self.env.config[k] = True
        term = {k: bool(self.env.config[k]) for k in TERMINATION_FLAGS}
        return {"ok": True, "env_create_s": time.time() - t0, "provenance": gen.provenance(), "pid": os.getpid(), "closed_loop": self.closed_loop,
                "termination": term, "horizon_guard": self.env.config["horizon"], "termination_keys": list(TERMINATION_KEYS),
                "note": "horizon_guard = generator make_env horizon (1000 env steps incl. the prefix); MetaDrive default None; engineering guard only, surfaces as native max_step"}

    def reset(self, req: dict[str, Any]) -> dict[str, Any]:
        from metadrive.component.traffic_participants.pedestrian import Pedestrian

        env = self.env
        self._clear_hazard()
        row = req.get("row") or {}
        # module factors (hazard body, FOV, settle steps, prefix throttle) from the saved row: v0.8 rows -> v0.8 values
        if row:
            _dist, prefix_throttle, _lat = gen.apply_row_factors(row)
        else:
            prefix_throttle = gen.PREFIX_THROTTLE
        if req.get("prefix_throttle") is not None:
            prefix_throttle = float(req["prefix_throttle"])
        seed, map_cfg, v0 = int(req["seed"]), str(req["map_cfg"]), float(req.get("v0", gen.V0_MPS))
        hazard = req.get("hazard")
        self.hazard = hazard
        # --- identical to run_episode up to the context frame --------------------------------------------------------
        gen.reset_scene(env, seed, map_cfg)
        if gen.SETTLE_STEPS <= 0:
            env.agent.set_velocity([1.0, 0.0], v0, in_local_frame=True)
        self.obj = None
        self.radius = 0.0
        spawn_pose = None
        if hazard is not None:
            pos, heading = gen.lane_pose(env, float(hazard["s_ahead"]) + float(hazard["prefix_travel"]), float(hazard["lateral"]))
            self.obj = gen.spawn_hazard_at(env, str(hazard["kind"]), pos, heading, bool(hazard["solid"]))
            self.radius = Pedestrian.RADIUS
            spawn_pose = [float(self.obj.position[0]), float(self.obj.position[1]), float(self.obj.origin.getZ()), float(self.obj.heading_theta)]
        if gen.SETTLE_STEPS > 0:
            gen.settle_physics(env, gen.SETTLE_STEPS)
            env.agent.set_velocity([1.0, 0.0], v0, in_local_frame=True)
        gen.pose_pedestrian_animation()
        states = [gen.ego_state(env)]
        flags: dict[str, list] = {k: [] for k in ("crash_human", "crash_object", "crash_sidewalk", "out_of_road", "center_dist", "lon", "lat", "gap", "hazard_z")}
        hazard_pose0 = [float(self.obj.position[0]), float(self.obj.position[1]), float(self.obj.origin.getZ()), float(self.obj.heading_theta)] if self.obj is not None else None
        obs = None
        for _t in range(gen.PREFIX_STEPS):
            gen.pose_pedestrian_animation()
            obs, _, _, _, info = env.step([0.0, prefix_throttle])
            states.append(gen.ego_state(env))
            a = env.agent
            flags["crash_human"].append(bool(a.crash_human)); flags["crash_object"].append(bool(a.crash_object))
            flags["crash_sidewalk"].append(bool(a.crash_sidewalk)); flags["out_of_road"].append(bool(info.get("out_of_road", False)))
            if self.obj is not None:
                dc, lon, lat, gap = gen.hazard_gap(env, self.obj, self.radius)
                flags["center_dist"].append(dc); flags["lon"].append(lon); flags["lat"].append(lat); flags["gap"].append(gap)
                flags["hazard_z"].append(float(self.obj.origin.getZ()))
        self.steps_after_context = 0
        # context lane frame for the progress measure
        self.ctx_lane = gen.ego_lane(env)
        self.ctx_pos = np.asarray(env.agent.position, dtype=np.float64)
        long_, lat_ = self.ctx_lane.local_coordinates(env.agent.position)
        self.ctx_long = float(long_)
        st = np.stack(states)
        out: dict[str, Any] = {
            "states": st.tolist(), **{k: v for k, v in flags.items() if v or k in ("crash_human", "crash_object", "crash_sidewalk", "out_of_road")},
            "hazard_pose": hazard_pose0, "hazard_spawn_pose": spawn_pose, "ctx_long": self.ctx_long, "ctx_lat": float(lat_),
            "prefix_travel_measured": gen.prefix_travel_of(st), "lane_length": float(self.ctx_lane.length),
            "ego_dims": [float(env.agent.LENGTH), float(env.agent.WIDTH)], "prefix_throttle": prefix_throttle,
            "factors": {"hazard_body": gen.HAZARD_BODY, "fov_deg": gen.FOV_DEG, "settle_steps": gen.SETTLE_STEPS, "prefix_steps": gen.PREFIX_STEPS},
        }
        out.update(self._capture(obs))
        return out

    def step(self, req: dict[str, Any]) -> dict[str, Any]:
        env = self.env
        chunk = np.asarray(req["chunk"], dtype=np.float64).reshape(-1, 2)
        raw = gen.raw_actions_from_chunks(chunk)
        want_frames = bool(req.get("want_frames", True))
        steps, frames, hmasks, cmasks, hpx = [], [], [], [], []
        done = False
        for t in range(raw.shape[0]):
            gen.pose_pedestrian_animation()
            obs, _, term, trunc, info = env.step([float(raw[t, 0]), float(raw[t, 1])])
            self.steps_after_context += 1
            rec = self._step_record(info, term, trunc)
            rec["steps_after_context"] = self.steps_after_context
            steps.append(rec)
            done = bool(self.closed_loop and (term or trunc))
            if (t + 1) % gen.SIM_STEPS_PER_MODEL_STEP == 0 or done:
                if want_frames:
                    cap = self._capture(obs)
                    frames.append(cap["frame"]); hmasks.append(cap["hazard_mask"]); cmasks.append(cap["corridor_mask"]); hpx.append(cap["hazard_px"])
            if done:
                break  # the env's own termination: stepping past done is not allowed without a reset
        return {"steps": steps, "done": done, "frames": frames, "hazard_mask": hmasks, "corridor_mask": cmasks, "hazard_px": hpx,
                "n_sim_steps": len(steps), "steps_after_context": self.steps_after_context}

    def close(self, req: dict[str, Any]) -> dict[str, Any]:
        self._clear_hazard()
        if self.env is not None:
            self.env.close()
            self.env = None
        return {"ok": True}


def main() -> None:
    sess = Session()
    handlers = {"init": sess.init, "reset": sess.reset, "step": sess.step, "close": sess.close}
    log(f"bridge up pid {os.getpid()} python {sys.version.split()[0]}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            t0 = time.time()
            resp = handlers[cmd](req)
            resp["_wall_s"] = time.time() - t0
        except Exception as exc:  # report, keep serving
            resp = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
            log(f"error on {line[:120]}: {exc}")
        _PROTO_OUT.write(json.dumps(resp) + "\n")
        _PROTO_OUT.flush()
        if req.get("cmd") == "close":
            break
    log("bridge exit")


if __name__ == "__main__":
    main()
