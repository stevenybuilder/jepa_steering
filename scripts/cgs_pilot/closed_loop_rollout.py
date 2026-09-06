#!/usr/bin/env python3
"""Closed-loop CEM adapter for the JEPA driving cell (cross model design jepa.md: Target hierarchy / Step 1 / C0 closed-loop gate).

Runs a frozen driving predictor with its own CEM planner CLOSED LOOP in MetaDrive from each factorial scene's context
state and labels every episode from the simulator:
    collision         a contact flag (crash_human | crash_object) fired
    safe_pass         no contact AND progress >= 0.9 x the hazard-free rollout's progress
    failure_progress  no contact, progress < 0.9 x reference (stopped / timeout)
``collision`` and ``failure_progress`` stay separate (an always-brake policy is a negative benchmark).

Design (see claude_handoff/closed_loop_adapter_report_2026-09-03.md for the full rationale):
- MetaDrive lives in another env without torch -> ``closed_loop_bridge.py`` subprocess (JSON lines on stdin/stdout).
- Branching from the context state = deterministic replay of the generator prefix (no Bullet save/restore); the replayed
  context state is checked against the stored cell (``ego_states[PREFIX_STEPS]``, <= 1e-6) or, when the cell npz is not
  on this box, against the manifest (bit-exact ``initial_state_sha256``, ``context_speed_mps``, ``hazard_pose_xyzh``,
  ``context_longitudinal_gap_m`` / ``context_lateral_m`` / ``prefix_travel_m`` <= 1e-6). The max deviation is stored in
  every rollout record (``replay``), and rollouts with ``replay_ok == false`` are excluded from the summary.
- Planner currency = steered_planner_ranking.py's progress goal: ``cost(a) = ||P(z_ctx, a)[-1] - z_goal||^2`` summed over
  the hazard U corridor tokens (token_groups.groups_from_driving_masks on the live context frame; hazard silhouette
  dilated by one patch). Closed-loop extension: the goal is ROLLING - z_goal at replan k is the encoded frame of the scene's
  hazard-free ALWAYS-THROTTLE rollout at model step (executed_steps + HORIZON), i.e. "be where the free-lane throttle
  rollout is, HORIZON imagined steps from now".
- CEM: the released jepa-wms CEM shape (Gaussian population, top-k elites re-fit, zero momentum, execute then replan) in the
  generator's chunk space: HORIZON model steps x [steer, throttle_brake] with steer fixed at 0 (the generator and the
  training clips use steer = 0 exactly; ``--steer-max`` > 0 is an exploratory, off-distribution option) and
  throttle_brake clipped to [--tb-min, --tb-max] = [-1, 0.5] (the generator's brake / throttle values). Defaults are
  compact (samples 64, elites 8, iterations 4, init N(0, 0.5); measured 5 ms per candidate-rollout on the shared 3090) instead of the released 300 x 15 because the search space
  is 3-D; all hyper-parameters are recorded in every record and in summary.json. The two generator chunks (brake, throttle)
  and the running mean are evaluated alongside every population as reference candidates (costs recorded: this gives the
  open-loop safe choice at every replan) but never enter the elite set. The executed action is the final CEM mean.
- Episode = up to --n-replans decisions, each executing --execute-steps model steps (default 1 = receding horizon; the
  released planner steps its whole 3-step plan, ``--execute-steps 3`` reproduces that), stopping at the first contact.

Label-first principle (cross model design jepa.md, BINDING; from COAST arXiv 2605.17144 s3.2 / s4.1): the simulator outcome
under the model's own planner is the label; rollouts for fitting and the C0 gate come from DISCOVERY scenes only - a seed
listed in a ``confirmation_seeds*.txt`` next to the stimulus (or in --confirmation-file) is refused unless
--allow-confirmation is given by a separate registered confirmation run. A model without a success/failure mix on the
solid in-lane identity is recorded as a C0 FAIL with its rates; the planner is never tuned to manufacture a mix (registered
remedies: earlier-epoch checkpoint of the same run, model scale, more training clips).

Per scene x arm physics (arm A: pedestrian solid / cone ghost; arm B: reverse), driven by that arm's model:
    hazard-free: always_throttle (goal frames + progress reference ``progress_ref_m``), always_brake, cem x episode seeds
    levels 0..3: always_throttle, always_brake, cem x episode seeds
Records: <out>/rollouts.jsonl (one line per episode; resumable), <out>/summary.json (rates per level with scene-bootstrap
95 % CIs, hazard-free goal success, always-brake rates, C0 gate, wall-clock).

Usage (model env; see run_closed_loop_c0.sh):
  closed_loop_rollout.py run --arm A --model-dir artifacts/drive_models/armA_seed0 --repo vendor/jepa-wms \
      --stimulus artifacts/drive_factorial_merged/armA --seeds-file .../discovery_seeds_common.txt --out artifacts/drive_closed_loop/armA_seed0
  closed_loop_rollout.py summarize --out artifacts/drive_closed_loop/armA_seed0
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import append_jsonl, sha256_array  # noqa: E402

IMG = 256
HORIZON = 3  # imagined / planned model steps (= the generator's clip horizon; metadrive_hazard_pilot.HORIZON)
PREFIX_STEPS = 5
SIM_STEPS_PER_MODEL_STEP = 3
HAZARD_LEVELS = {0: ("ped", "sidewalk_right"), 1: ("ped", "in_lane"), 2: ("ped", "adjacent_lane_left"), 3: ("cone", "in_lane")}
LEVEL_NAMES = {0: "H0_ped_sidewalk", 1: "H1_ped_in_lane", 2: "H0p_ped_mirror_null", 3: "H1_obj_cone_in_lane"}
ARMS = {"A": {"ped": True, "cone": False}, "B": {"ped": False, "cone": True}}
SOLID_LEVEL = {"A": 1, "B": 3}
GHOST_LEVEL = {"A": 3, "B": 1}
BRAKE_CHUNK = (0.0, -1.0)
THROTTLE_CHUNK = (0.0, 0.5)
POLICIES = ("always_throttle", "always_brake", "cem")
BENCH_EXEC_STEPS = 10  # native mode: benchmark policies (no decisions) step 10 model steps per bridge call
NATIVE_LABELS = ("success", "failure")
MIN_MIX = 3  # COAST App. A.8 eligibility: >= 3 successes and >= 3 failures per model x level
LABELS = ("collision", "safe_pass", "failure_progress")
REPLAY_TOL = 1e-6
DIVERGENCE_TOL = 0.1  # executed throttle_brake differs from the reference chunk at the same replan index by more than this (range is [-1, 0.5])
PASS_FRACTION = 0.9
C0_SAFE_PASS_BAND = (0.20, 0.90)
C0_FREE_SUCCESS_MIN = 0.50


# ----------------------------------------------------------------------------- bridge client


class Bridge:
    """Subprocess client for closed_loop_bridge.py (MetaDrive env)."""

    def __init__(self, python: str, code_dir: Path, log_path: Path) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        env["PYTHONUNBUFFERED"] = "1"
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen([python, str(code_dir / "closed_loop_bridge.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, cwd=str(code_dir), env=env, text=True, bufsize=1)
        self.wall = {"reset": 0.0, "step": 0.0, "n_reset": 0, "n_step": 0}

    def call(self, req: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        t0 = time.time()
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"bridge died (rc={self.proc.poll()}); see {self.log.name}")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"bridge error for {req.get('cmd')}: {resp['error']}\n{resp.get('traceback', '')}")
        cmd = req.get("cmd")
        if cmd in ("reset", "step"):
            self.wall[cmd] += time.time() - t0
            self.wall[f"n_{cmd}"] += 1
        return resp

    def close(self) -> None:
        try:
            self.call({"cmd": "close"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        self.log.close()


def decode_frame(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(IMG, IMG, 3)


def decode_mask(s: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))[: IMG * IMG]
    return bits.reshape(IMG, IMG).astype(bool)


# ----------------------------------------------------------------------------- stimulus


def load_scenes(stimulus: Path, arm: str) -> dict[int, dict[str, Any]]:
    """{seed: {"rows": {(hazard, action): row}, "dir": Path}} from a merged manifest (``<stimulus>/manifest.jsonl``, artifact
    paths absolute or relative to <stimulus>) or a raw factorial arm dir (``<stimulus>/seed_*/manifest.jsonl``)."""
    files = [(stimulus, stimulus / "manifest.jsonl")] if (stimulus / "manifest.jsonl").exists() else \
        [(p.parent, p) for p in sorted(stimulus.glob("seed_*/manifest.jsonl"))]
    if not files:
        raise SystemExit(f"no manifest.jsonl under {stimulus}")
    scenes: dict[int, dict[str, Any]] = {}
    for d, f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("arm", arm)) != arm:
                continue
            s = scenes.setdefault(int(r["seed"]), {"rows": {}, "dir": d})
            s["rows"][(int(r["hazard"]), int(r["candidate_action"]))] = r
    return scenes


def cell_path(scene: dict[str, Any], row: dict[str, Any]) -> Path | None:
    p = Path(row["artifact"])
    if not p.is_absolute():
        p = scene["dir"] / p
    return p if p.exists() else None


def hazard_spec(row0: dict[str, Any], level: int, arm: str, lateral_by_level: dict[int, float]) -> dict[str, Any]:
    kind, _where = HAZARD_LEVELS[level]
    return {"kind": kind, "s_ahead": float(row0["hazard_dist_m"]), "lateral": float(lateral_by_level[level]),
            "solid": bool(ARMS[arm][kind]), "prefix_travel": float(row0["prefix_travel_m"])}


# ----------------------------------------------------------------------------- replay verification


def verify_replay(resp: dict[str, Any], row: dict[str, Any] | None, cell: Path | None, level: int | None) -> dict[str, Any]:
    """Compare the replayed prefix with the stored cell / manifest row of the same (scene, level); ``ok`` = every available
    check within REPLAY_TOL (the ego_states check when the npz is present, the manifest checks otherwise)."""
    st = np.asarray(resp["states"], dtype=np.float64)
    out: dict[str, Any] = {"source": None, "ctx_state_max_dev": None, "initial_state_max_dev": None, "initial_state_sha_match": None,
                           "ctx_speed_dev": None, "ctx_gap_dev": None, "ctx_lateral_dev": None, "hazard_pose_dev": None,
                           "prefix_travel_dev": None, "ctx_frame_diff_px": None, "ctx_frame_max_abs": None, "ok": None}
    if row is None:
        out["ok"] = None
        return out
    checks: list[float] = []
    if cell is not None:
        with np.load(cell) as c:
            saved = np.asarray(c["ego_states"], dtype=np.float64)
            ctx_frame = np.asarray(c["context_frames"])[0]
        out["source"] = "cell_npz"
        out["ctx_state_max_dev"] = float(np.abs(st[PREFIX_STEPS] - saved[PREFIX_STEPS]).max())
        out["initial_state_max_dev"] = float(np.abs(st[0] - saved[0]).max())
        fd = np.abs(decode_frame(resp["frame"]).astype(np.int16) - ctx_frame.astype(np.int16))
        out["ctx_frame_diff_px"] = int((fd.max(-1) > 0).sum())
        out["ctx_frame_max_abs"] = int(fd.max())
        checks += [out["ctx_state_max_dev"], out["initial_state_max_dev"]]
    else:
        out["source"] = "manifest_only"
    out["initial_state_sha_match"] = bool(sha256_array(st[0]) == row.get("initial_state_sha256"))
    out["ctx_speed_dev"] = float(abs(st[PREFIX_STEPS, 6] - float(row["context_speed_mps"])))
    checks.append(out["ctx_speed_dev"])
    out["prefix_travel_dev"] = float(abs(float(resp["prefix_travel_measured"]) - float(row["prefix_travel_m"])))
    checks.append(out["prefix_travel_dev"])
    if level is not None and resp.get("hazard_pose") is not None:
        out["hazard_pose_dev"] = float(np.abs(np.asarray(resp["hazard_pose"], dtype=np.float64) - np.asarray(row["hazard_pose_xyzh"], dtype=np.float64)).max())
        out["ctx_gap_dev"] = float(abs(resp["lon"][PREFIX_STEPS - 1] - float(row["context_longitudinal_gap_m"])))
        out["ctx_lateral_dev"] = float(abs(resp["lat"][PREFIX_STEPS - 1] - float(row["context_lateral_m"])))
        checks += [out["hazard_pose_dev"], out["ctx_gap_dev"], out["ctx_lateral_dev"]]
    out["max_dev"] = float(max(checks)) if checks else None
    out["ok"] = bool(out["initial_state_sha_match"] and all(c <= REPLAY_TOL for c in checks))
    return out


# ----------------------------------------------------------------------------- model side


class Planner:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from model_action_sensitivity import encode_frames, load_model
        from token_groups import groups_from_driving_masks

        self.torch = torch
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model(args.repo, args.model_dir / "eval_config.yaml", args.model_dir / args.checkpoint, args.model_name, str(self.device))
        self._encode, self._groups = encode_frames, groups_from_driving_masks
        self.cfg = {"samples": args.cem_samples, "elites": args.cem_elites, "iters": args.cem_iters, "init_mean": args.cem_init_mean,
                    "init_std": args.cem_init_std, "min_std": args.cem_min_std, "tb_min": args.tb_min, "tb_max": args.tb_max,
                    "steer_max": args.steer_max, "horizon": HORIZON, "eval_batch": args.eval_batch, "token_group": "hazard_corridor",
                    "hazard_dilate": 1, "executed": "final_mean", "momentum": 0.0}
        self.wall = {"encode": 0.0, "unroll": 0.0, "n_unroll_cands": 0}
        self._goal_cache: dict[str, Any] = {}

    def encode(self, frame: np.ndarray):
        t0 = time.time()
        z = self._encode(self.model, frame[None], self.device)
        self.wall["encode"] += time.time() - t0
        return z

    def goal_latent(self, key: str, frame: np.ndarray):
        if key not in self._goal_cache:
            z = self.encode(frame)
            self._goal_cache[key] = z.reshape(-1, z.shape[-1]).float()  # [n_tokens, D]
        return self._goal_cache[key]

    def clear_goal_cache(self) -> None:
        self._goal_cache = {}

    def tokens(self, hazard_mask: np.ndarray, corridor_mask: np.ndarray) -> np.ndarray:
        g = self._groups({"hazard_mask": hazard_mask[None], "corridor_mask": corridor_mask[None]}, hazard_dilate=self.cfg["hazard_dilate"])
        return np.asarray(g.group(0, self.cfg["token_group"]), dtype=np.int64)

    def costs(self, z_ctx, cands: np.ndarray, z_goal, idx: np.ndarray) -> np.ndarray:
        """cands [n, H, 2] -> [n] L2 costs on the token set of the LAST predicted frame."""
        torch = self.torch
        out = []
        idx_t = torch.as_tensor(idx, device=self.device)
        goal = z_goal[idx_t]
        t0 = time.time()
        with torch.inference_mode():
            for i in range(0, cands.shape[0], self.cfg["eval_batch"]):
                c = torch.from_numpy(np.ascontiguousarray(cands[i:i + self.cfg["eval_batch"]])).to(device=self.device, dtype=torch.float32)
                act = c.permute(1, 0, 2).contiguous()  # [H, B, 2]
                roll = self.model.unroll(z_ctx, act_suffix=act)
                last = roll[-1]
                last = last.reshape(last.shape[0], -1, last.shape[-1]).float()  # [B, n_tokens, D]
                out.append(((last[:, idx_t] - goal[None]) ** 2).sum(dim=(1, 2)).cpu().numpy().astype(np.float64))
        self.wall["unroll"] += time.time() - t0
        self.wall["n_unroll_cands"] += int(cands.shape[0])
        return np.concatenate(out)

    def plan(self, z_ctx, z_goal, idx: np.ndarray, rng: np.random.RandomState) -> tuple[np.ndarray, dict[str, Any]]:
        cfg = self.cfg
        H = cfg["horizon"]
        mu = np.full(H, cfg["init_mean"], dtype=np.float64)
        sigma = np.full(H, cfg["init_std"], dtype=np.float64)
        steer_on = cfg["steer_max"] > 0
        mu_s = np.zeros(H); sigma_s = np.full(H, cfg["steer_max"] / 2 if steer_on else 0.0)
        brake = np.repeat(np.asarray([BRAKE_CHUNK]), H, axis=0)
        throttle = np.repeat(np.asarray([THROTTLE_CHUNK]), H, axis=0)
        hist = []
        for it in range(cfg["iters"]):
            tb = np.clip(rng.randn(cfg["samples"], H) * sigma + mu, cfg["tb_min"], cfg["tb_max"])
            st = np.clip(rng.randn(cfg["samples"], H) * sigma_s + mu_s, -cfg["steer_max"], cfg["steer_max"]) if steer_on else np.zeros((cfg["samples"], H))
            samples = np.stack([st, tb], axis=-1)  # [n, H, 2]
            mean_chunk = np.stack([np.clip(mu_s, -cfg["steer_max"], cfg["steer_max"]) if steer_on else np.zeros(H), np.clip(mu, cfg["tb_min"], cfg["tb_max"])], axis=-1)
            batch = np.concatenate([samples, brake[None], throttle[None], mean_chunk[None]], axis=0).astype(np.float32)
            cost = self.costs(z_ctx, batch, z_goal, idx)
            cs, c_brake, c_throttle, c_mean = cost[: cfg["samples"]], float(cost[-3]), float(cost[-2]), float(cost[-1])
            elite = np.argsort(cs)[: cfg["elites"]]
            mu = tb[elite].mean(axis=0)
            sigma = np.maximum(tb[elite].std(axis=0), cfg["min_std"])
            if steer_on:
                mu_s = st[elite].mean(axis=0); sigma_s = np.maximum(st[elite].std(axis=0), cfg["min_std"])
            hist.append({"iter": it, "cost_min": float(cs.min()), "cost_mean": float(cs.mean()), "cost_max": float(cs.max()),
                         "cost_elite_mean": float(cs[elite].mean()), "cost_brake": c_brake, "cost_throttle": c_throttle, "cost_mean_chunk": c_mean,
                         "mu_tb": mu.round(4).tolist(), "sigma_tb": sigma.round(4).tolist(), **({"mu_steer": mu_s.round(4).tolist()} if steer_on else {})})
        final = np.stack([np.clip(mu_s, -cfg["steer_max"], cfg["steer_max"]) if steer_on else np.zeros(H), np.clip(mu, cfg["tb_min"], cfg["tb_max"])], axis=-1)
        c_final = float(self.costs(z_ctx, final[None].astype(np.float32), z_goal, idx)[0])
        last = hist[-1]
        info = {"n_tokens": int(idx.size), "cost_final": c_final, "cost_brake": last["cost_brake"], "cost_throttle": last["cost_throttle"],
                "delta_open_loop": last["cost_throttle"] - last["cost_brake"], "safe_choice_open_loop": bool(last["cost_throttle"] > last["cost_brake"]),
                "cost_min_pop": min(h["cost_min"] for h in hist), "iters": hist}
        return final, info


# ----------------------------------------------------------------------------- episodes


def episode_rng(model_tag: str, seed: int, level: int | None, episode_seed: int) -> np.random.RandomState:
    d = hashlib.sha256(f"{model_tag}:{seed}:{'free' if level is None else level}:{episode_seed}".encode()).digest()
    return np.random.RandomState(int.from_bytes(d[:4], "big"))


def run_one(bridge: Bridge, planner: Planner | None, args: argparse.Namespace, scene: dict[str, Any], seed: int, level: int | None,
            policy: str, episode_seed: int, goal_frames: list[np.ndarray] | None, n_model_steps: int,
            lateral_by_level: dict[int, float], model_tag: str, collect_frames: list[np.ndarray] | None = None) -> dict[str, Any]:
    """One closed-loop episode from the scene's context state. Returns the rollout record (progress-rule labels filled in
    later by run_scene). ``collect_frames`` (list) receives the context frame and the frame after every executed model step.
    label_mode 'native': run until the env's own done (MetaDrive default termination + the horizon guard), outcome from the
    native predicates; label_mode 'progress_rule' (legacy): n_model_steps or first contact."""
    native = args.label == "native"
    row0 = scene["rows"][(0, 0)]
    row = scene["rows"].get((level, 0)) if level is not None else row0
    hz = None if level is None else hazard_spec(row0, level, args.arm, lateral_by_level)
    t_ep = time.time()
    resp = bridge.call({"cmd": "reset", "seed": seed, "map_cfg": str(row0["map_cfg"]), "row": row0, "v0": float(row0.get("v0_mps", 8.0)), "hazard": hz})
    # the hazard-free context state equals the hazard cells' context state to <= 1e-9 (no contact in the prefix; generator gate)
    replay = verify_replay(resp, row, cell_path(scene, row) if row is not None else None, level)
    frame = decode_frame(resp["frame"])
    hmask, cmask = decode_mask(resp["hazard_mask"]), decode_mask(resp["corridor_mask"])
    if collect_frames is not None:
        collect_frames.append(frame)
    ctx_state = list(map(float, resp["states"][PREFIX_STEPS]))
    steps_rec: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    executed = 0
    contact_step = None
    rng = episode_rng(model_tag, seed, level, episode_seed)
    exec_steps = args.execute_steps if policy == "cem" else (BENCH_EXEC_STEPS if native else args.execute_steps)
    cap_model_steps = args.max_model_steps if native else n_model_steps
    n_replans = int(np.ceil(cap_model_steps / exec_steps))
    want_frames = policy == "cem" or collect_frames is not None
    z_list: list[np.ndarray] = []  # encoded context latent per replan (CEM only), saved to <out>/latents/
    tok_list: list[np.ndarray] = []
    env_done = False
    for k in range(n_replans):
        rec: dict[str, Any] = {"replan": k, "model_steps_before": executed, "sim_step_before": executed * SIM_STEPS_PER_MODEL_STEP}
        if policy == "cem":
            assert planner is not None and goal_frames is not None
            gi = min(executed + HORIZON, len(goal_frames) - 1)
            z_ctx = planner.encode(frame)
            z_list.append(z_ctx.reshape(-1, z_ctx.shape[-1]).float().cpu().numpy().astype(np.float16))
            z_goal = planner.goal_latent(f"{seed}:{gi}", goal_frames[gi])
            idx = planner.tokens(hmask, cmask)
            if idx.size == 0:
                idx = np.arange(IMG // 16 * IMG // 16)
                rec["tokens_fallback_all"] = True
            chunk, info = planner.plan(z_ctx, z_goal, idx, rng)
            tm = np.zeros(z_list[-1].shape[0], dtype=bool); tm[idx] = True; tok_list.append(tm)
            rec.update({"goal_index": gi, "cem": info, "n_tokens": int(idx.size), "n_hazard_patches": int(planner._groups({"hazard_mask": hmask[None], "corridor_mask": cmask[None]}).group(0, "hazard").size)})
            n_exec = min(exec_steps, cap_model_steps - executed)
            exec_chunk = np.asarray(chunk[:n_exec], dtype=np.float64)
        elif policy in ("always_throttle", "always_brake"):
            base = THROTTLE_CHUNK if policy == "always_throttle" else BRAKE_CHUNK
            chunk = np.repeat(np.asarray([base]), HORIZON, axis=0)
            n_exec = min(exec_steps, cap_model_steps - executed)
            exec_chunk = np.repeat(np.asarray([base], dtype=np.float64), n_exec, axis=0)
        else:
            raise ValueError(policy)
        s = bridge.call({"cmd": "step", "chunk": exec_chunk.tolist(), "want_frames": want_frames})
        n_done = int(np.ceil(s["n_sim_steps"] / SIM_STEPS_PER_MODEL_STEP))
        executed += n_done
        last = s["steps"][-1]
        rec.update({"chosen_chunk": np.asarray(chunk, dtype=np.float64).round(5).tolist(), "executed_chunk": exec_chunk[:n_done].round(5).tolist(),
                    "n_sim_steps": int(s["n_sim_steps"]), "state_after": last["state"], "speed_after": last["speed"], "progress_after_m": last.get("progress_m"),
                    "gap_after_m": last.get("gap"), "min_gap_m": (min(x["gap"] for x in s["steps"]) if "gap" in last else None),
                    "contact": bool(any(x["crash_human"] or x["crash_object"] for x in s["steps"])),
                    "crash_sidewalk": bool(any(x["crash_sidewalk"] for x in s["steps"])), "out_of_road": bool(any(x["out_of_road"] for x in s["steps"])),
                    "done": bool(s.get("done", False)), "route_completion_after": last.get("route_completion")})
        for j, x in enumerate(s["steps"]):
            steps_rec.append({"sim_step": rec["sim_step_before"] + j + 1, **{kk: x[kk] for kk in x if kk != "state"}})
            if contact_step is None and (x["crash_human"] or x["crash_object"]):
                contact_step = rec["sim_step_before"] + j + 1
        replans.append(rec)
        if collect_frames is not None:
            collect_frames.extend(decode_frame(f) for f in s["frames"])
        if s["frames"]:
            frame = decode_frame(s["frames"][-1])
            hmask, cmask = decode_mask(s["hazard_mask"][-1]), decode_mask(s["corridor_mask"][-1])
        env_done = bool(s.get("done", False))
        if env_done or executed >= cap_model_steps:
            break
        if not native and contact_step is not None:
            break
    final = steps_rec[-1]
    latent_path = None
    if z_list:
        (args.out / "latents").mkdir(parents=True, exist_ok=True)
        lp = args.out / "latents" / f"{row0['pair_id']}__l{'free' if level is None else level}__{policy}__e{episode_seed}.npz"
        np.savez_compressed(lp, z_context=np.stack(z_list), token_mask=np.stack(tok_list), goal_index=np.asarray([r["goal_index"] for r in replans]),
                            executed_tb=np.asarray([r["executed_chunk"][0][1] for r in replans]), model_steps_before=np.asarray([r["model_steps_before"] for r in replans]),
                            meta=np.array(json.dumps({"model": model_tag, "seed": seed, "level": level, "policy": policy, "episode_seed": episode_seed,
                                                       "layout": "z_context [n_replans, n_tokens, D] fp16 = frozen-encoder latent of the context frame at each replan; token_mask [n_replans, n_tokens] = hazard U corridor tokens used by the cost"})))
        latent_path = str(lp.relative_to(args.out))
    contact_replan = next((r["replan"] for r in replans if r["contact"]), None)
    brake_replan = next((r["replan"] for r in replans if float(r["executed_chunk"][0][1]) < 0.0), None)
    out = {
        "model": model_tag, "arm": args.arm, "pair_id": row0["pair_id"], "seed": seed, "map_cfg": str(row0["map_cfg"]), "level": level,
        "level_name": (LEVEL_NAMES[level] if level is not None else "hazard_free"),
        "hazard_kind": (HAZARD_LEVELS[level][0] if level is not None else None), "solid": (bool(ARMS[args.arm][HAZARD_LEVELS[level][0]]) if level is not None else None),
        "policy": policy, "episode_seed": episode_seed, "hazard": hz, "hazard_pose": resp.get("hazard_pose"), "label_mode": args.label,
        "context_state": ctx_state, "context_speed_mps": ctx_state[6], "context_gap_m": (resp["gap"][PREFIX_STEPS - 1] if hz else None),
        "context_center_distance_m": (resp["center_dist"][PREFIX_STEPS - 1] if hz else None),
        "replay": replay, "replay_ok": replay["ok"], "replay_max_dev": replay.get("max_dev"),
        "n_model_steps_planned": (None if native else n_model_steps), "max_model_steps_guard": (cap_model_steps if native else None),
        "n_model_steps_executed": executed, "execute_steps": exec_steps, "n_replans": len(replans),
        "contact": contact_step is not None, "contact_step": contact_step, "contact_model_step": (int(np.ceil(contact_step / SIM_STEPS_PER_MODEL_STEP)) if contact_step else None),
        "crash_human": bool(any(x["crash_human"] for x in steps_rec)), "crash_object": bool(any(x["crash_object"] for x in steps_rec)),
        "crash_sidewalk": bool(any(x["crash_sidewalk"] for x in steps_rec)), "out_of_road": bool(any(x["out_of_road"] for x in steps_rec)),
        "progress_m": float(final["progress_m"]), "disp_m": float(final["disp_m"]), "final_speed_mps": float(final["speed"]), "final_lane_lat_m": float(final["lane_lat_m"]),
        "min_gap_m": (min(x["gap"] for x in steps_rec) if hz else None), "min_center_distance_m": (min(x["center_dist"] for x in steps_rec) if hz else None),
        "progress_per_model_step_m": [x["progress_m"] for x in steps_rec if x["sim_step"] % SIM_STEPS_PER_MODEL_STEP == 0],
        "replans": replans, "wall_s": time.time() - t_ep, "latent_path": latent_path,
        "first_contact_replan_idx": contact_replan, "first_brake_replan_idx": brake_replan,
        "divergence_replan_idx": None, "divergence_reference": None, "chunk_dev_per_replan": None, "divergence_tol": DIVERGENCE_TOL,
        "divergence_replan_idx_vs_throttle": first_divergence([r["executed_chunk"] for r in replans], [[list(THROTTLE_CHUNK)] * exec_steps] * len(replans), DIVERGENCE_TOL)[0],
        "label": None, "label_ref": None, "progress_ref_m": None, "progress_free_cem_m": None,
    }
    if native:
        out.update(native_outcome(steps_rec, env_done, executed >= cap_model_steps and not env_done))
        for k in ("progress_m", "disp_m", "final_speed_mps", "final_lane_lat_m", "progress_per_model_step_m", "min_gap_m", "min_center_distance_m"):
            out["descriptive_" + k] = out.pop(k)
        out["label"] = out["outcome"]
    return out


NATIVE_CRASH_KEYS = ("crash_human", "crash_object", "crash_vehicle", "crash_building", "crash_sidewalk")
FAILURE_PRIORITY = ("crash_human", "crash_object", "crash_vehicle", "crash_building", "crash_sidewalk", "out_of_road", "max_step", "driver_cap")


def native_outcome(steps_rec: list[dict[str, Any]], env_done: bool, hit_cap: bool) -> dict[str, Any]:
    """Label-first principle, amendment 1: the simulator's NATIVE predicates only. success = info['arrive_dest'] at the env's
    own done with no crash predicate at any earlier or the same step; otherwise failure with the native reason as subtype."""
    infos = [x["done_info"] for x in steps_rec]
    any_flags = {k: bool(any(d.get(k, False) for d in infos)) for k in infos[0]}
    final = dict(steps_rec[-1]["done_info"])
    first_step = {k: next((x["sim_step"] for x, d in zip(steps_rec, infos) if d.get(k, False)), None) for k in infos[0]}
    crash_any = any(any_flags.get(k, False) for k in NATIVE_CRASH_KEYS)
    arrive = bool(final.get("arrive_dest", False))
    same_step = bool(arrive and any(final.get(k, False) for k in NATIVE_CRASH_KEYS))
    success = bool(env_done and arrive and not crash_any)
    reason = None
    if not success:
        reason = next((k for k in FAILURE_PRIORITY if (k == "driver_cap" and hit_cap) or any_flags.get(k, False)), None)
        if reason is None:
            reason = "driver_cap" if hit_cap else ("env_done_without_predicate" if env_done else "unterminated")
    return {"outcome": ("success" if success else "failure"), "failure_reason": reason, "native_done_info_final": final, "native_done_info_any": any_flags,
            "native_first_step": first_step, "native_route_completion": steps_rec[-1].get("route_completion"),
            "env_steps_after_context": int(steps_rec[-1].get("steps_after_context", len(steps_rec))), "env_episode_step_final": int(steps_rec[-1].get("env_episode_step", -1)),
            "terminated": bool(steps_rec[-1].get("terminated", False)), "truncated": bool(steps_rec[-1].get("truncated", False)), "env_done": bool(env_done),
            "hit_driver_cap": bool(hit_cap), "arrive_dest_env_step": first_step.get("arrive_dest"), "success_and_crash_same_step": same_step}


def first_divergence(chunks: list, ref_chunks: list, tol: float = DIVERGENCE_TOL) -> tuple[int | None, list[float]]:
    """First replan index at which the executed chunk's throttle_brake values differ from the reference rollout's executed
    chunk at the same index by more than ``tol`` (max abs over the executed model steps; steer compared too). A reference
    shorter than the episode (it ended early) counts as divergence at its end. Returns (index | None, per-replan max dev)."""
    devs: list[float] = []
    for k, c in enumerate(chunks):
        if k >= len(ref_chunks):
            devs.append(float("inf"))
            continue
        a, b = np.asarray(c, dtype=np.float64), np.asarray(ref_chunks[k], dtype=np.float64)
        n = min(a.shape[0], b.shape[0])
        devs.append(float(np.abs(a[:n] - b[:n]).max()))
    idx = next((k for k, d in enumerate(devs) if d > tol), None)
    return idx, [d if np.isfinite(d) else None for d in devs]


STRONG_BRAKE_TB = -0.5  # executed throttle_brake at or below this = an avoidance response (the generator's brake chunk is -1)


def first_strong_brake(rec: dict[str, Any], tb: float = STRONG_BRAKE_TB) -> int | None:
    """First replan whose executed first-step throttle_brake <= ``tb`` (post-hoc, from the stored executed chunks)."""
    return next((r["replan"] for r in rec.get("replans") or [] if float(r["executed_chunk"][0][1]) <= tb), None)


def truncation_mask(rec: dict[str, Any], tol: float | None = None, strong_brake: float | None = None) -> list[int]:
    """Replan indices of ``rec`` that are strictly PRE-OUTCOME: before the first divergence from the hazard-free reference
    rollout (``divergence_replan_idx``; recomputed from ``chunk_dev_per_replan`` when ``tol`` is given) and before the first
    contact. An episode that never diverges and never touches keeps every replan (its whole path is pre-outcome).
    ``strong_brake`` (throttle_brake threshold, e.g. STRONG_BRAKE_TB) replaces the chunk-deviation divergence by the first
    strong brake: the smoke test showed the progress currency is nearly flat between coast and throttle, so the hazard-free
    reference's own CEM chunks jitter by up to ~0.5 and the 0.1-tolerance divergence fires at index 0 without any hazard
    response; the raw deviations stay in the record for any other rule."""
    n = len(rec.get("replans") or [])
    cut = n
    if strong_brake is not None:
        div = first_strong_brake(rec, strong_brake)
    else:
        div = rec.get("divergence_replan_idx")
        if tol is not None and rec.get("chunk_dev_per_replan") is not None:
            div = next((k for k, d in enumerate(rec["chunk_dev_per_replan"]) if d is None or d > tol), None)
    for v in (div, rec.get("first_contact_replan_idx")):
        if v is not None:
            cut = min(cut, int(v))
    return list(range(cut))


def progress_of(rec: dict[str, Any]) -> float:
    return float(rec["progress_m"] if "progress_m" in rec else rec["descriptive_progress_m"])


def label_of(rec: dict[str, Any], denominator: float | None) -> str | None:
    if rec["contact"]:
        return "collision"
    if denominator is None:
        return None
    return "safe_pass" if progress_of(rec) >= PASS_FRACTION * denominator else "failure_progress"


def sealed_seeds(stimulus: Path, extra: list[Path] | None) -> set[int]:
    """Confirmation (sealed) seeds: every confirmation_seeds*.txt next to the stimulus (and its parent) plus --confirmation-file."""
    files = list(stimulus.glob("confirmation_seeds*.txt")) + list(stimulus.parent.glob("confirmation_seeds*.txt")) + list(extra or [])
    out: set[int] = set()
    for f in files:
        if f.exists():
            out |= {int(t) for t in f.read_text().split() if t.strip().lstrip("-").isdigit()}
    return out


def done_keys(out: Path) -> set[tuple]:
    f = out / "rollouts.jsonl"
    keys = set()
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((int(r["seed"]), r["level"], r["policy"], int(r["episode_seed"])))
    return keys


def run_scene(bridge: Bridge, planner: Planner | None, args: argparse.Namespace, scene: dict[str, Any], seed: int, model_tag: str, done: set[tuple], log) -> list[dict[str, Any]]:
    native = args.label == "native"
    row0 = scene["rows"][(0, 0)]
    lateral_by_level = {lvl: float(scene["rows"][(lvl, 0)]["hazard_lateral_m"]) for lvl in (0, 1, 2, 3)}
    n_model_steps = args.n_replans * args.execute_steps
    out_rows: list[dict[str, Any]] = []
    t0 = time.time()
    # 1. hazard-free ALWAYS-THROTTLE reference: goal frames + progress reference; native mode: runs to the env's own done and
    #    must reach arrive_dest (reachability check; otherwise the scene is refused, never re-goaled)
    goal_frames: list[np.ndarray] = []  # goal frame j = frame after j hazard-free throttle model steps (index 0 = context frame)
    free_thr = run_one(bridge, None, args, scene, seed, None, "always_throttle", 0, None, n_model_steps + HORIZON, lateral_by_level, model_tag, collect_frames=goal_frames)
    prog = free_thr["progress_per_model_step_m" if not native else "descriptive_progress_per_model_step_m"]
    progress_ref = float(prog[n_model_steps - 1]) if len(prog) >= n_model_steps else progress_of(free_thr)
    if native:
        free_thr["reference_reachable"] = bool(free_thr["outcome"] == "success")
        free_thr["note"] = (f"hazard-free always-throttle reference: arrive_dest at env step {free_thr['arrive_dest_env_step']} after the context "
                            f"(horizon guard {args.max_model_steps * SIM_STEPS_PER_MODEL_STEP} model-step-equivalent env steps); descriptive progress reference = progress at model step {n_model_steps}")
        if not free_thr["reference_reachable"]:
            msg = f"REFERENCE_UNREACHABLE seed {seed} map {row0['map_cfg']}: hazard-free always-throttle ended with {free_thr['failure_reason']} (done_info {free_thr['native_done_info_final']}); scene refused (no goal is invented)"
            print(msg, file=log, flush=True)
            append_jsonl(args.out / "skipped_scenes.jsonl", {"seed": seed, "map_cfg": row0["map_cfg"], "reason": msg, "reference": json_safe({k: v for k, v in free_thr.items() if k != "replans"})})
            if args.stop_on_unreachable:
                raise SystemExit(msg)
            return []
    ref_label = "collision" if free_thr["contact"] else "safe_pass"  # progress-rule label of the reference itself
    free_thr.update({"progress_ref_m": progress_ref, "n_model_steps_reference": n_model_steps, "label_ref": ref_label, "n_goal_frames": len(goal_frames)})
    if native:
        free_thr.update({"descriptive_progress_ref_m": free_thr.pop("progress_ref_m"), "descriptive_label_ref_progress_rule": free_thr.pop("label_ref")})
    else:
        free_thr["label"] = ref_label
        free_thr["note"] = f"hazard-free reference: progress at model step {n_model_steps} = {progress_ref:.3f} m; ran {n_model_steps + HORIZON} steps for the rolling goal frames"
    if planner is not None:
        planner.clear_goal_cache()
    if (seed, None, "always_throttle", 0) not in done:
        out_rows.append(free_thr)
    free_cem: dict[int, float] = {}
    ref_chunks: dict[tuple[str, int], list] = {("always_throttle", 0): [r["executed_chunk"] for r in free_thr["replans"]]}
    plan: list[tuple[int | None, str, int]] = [(None, "always_brake", 0)]
    for e in range(args.episode_seeds):
        plan.append((None, "cem", e))
    for lvl in (0, 1, 2, 3):
        plan += [(lvl, "always_throttle", 0), (lvl, "always_brake", 0)] + [(lvl, "cem", e) for e in range(args.episode_seeds)]
    if (args.out / "rollouts.jsonl").exists():  # denominators / reference chunks of already-finished hazard-free episodes (resume)
        for line in (args.out / "rollouts.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if int(r["seed"]) == seed and r["level"] is None:
                    ref_chunks[(r["policy"], int(r["episode_seed"]))] = [x["executed_chunk"] for x in r["replans"]]
                    if r["policy"] == "cem":
                        free_cem[int(r["episode_seed"])] = progress_of(r)
    for lvl, policy, e in plan:
        if (seed, lvl, policy, e) in done:
            continue
        if policy == "cem" and planner is None:
            continue
        rec = run_one(bridge, planner, args, scene, seed, lvl, policy, e, goal_frames if policy == "cem" else None, n_model_steps, lateral_by_level, model_tag)
        rec["progress_ref_m"] = progress_ref
        rec["label_ref"] = label_of(rec, progress_ref)
        if lvl is None:
            ref_chunks[(policy, e)] = [x["executed_chunk"] for x in rec["replans"]]
            if policy == "cem":
                free_cem[e] = progress_of(rec)
                rec["progress_free_cem_m"] = progress_of(rec)
                rec["free_goal_success"] = bool((not rec["contact"]) and (not rec["out_of_road"]) and progress_of(rec) >= PASS_FRACTION * progress_ref)
            rec["label_progress_rule"] = rec["label_ref"]
        else:
            den = free_cem.get(e if policy == "cem" else 0)
            rec["progress_free_cem_m"] = den
            rec["label_progress_rule"] = label_of(rec, den)
            rec["label_denominator"] = "hazard_free_cem_same_episode_seed" if policy == "cem" else "hazard_free_cem_episode_seed_0"
            ref = ref_chunks.get((policy, e if policy == "cem" else 0))
            if ref is not None:
                rec["divergence_replan_idx"], rec["chunk_dev_per_replan"] = first_divergence([x["executed_chunk"] for x in rec["replans"]], ref, DIVERGENCE_TOL)
                rec["divergence_reference"] = f"hazard_free_{policy}_episode_seed_{e if policy == 'cem' else 0}"
            rec["pre_outcome_replans"] = truncation_mask(rec)
        if native:  # our progress/goal quantities are DESCRIPTIVE only (amendment 1); the label is the native outcome
            for k in ("progress_ref_m", "label_ref", "progress_free_cem_m", "free_goal_success", "label_progress_rule"):
                if k in rec:
                    rec["descriptive_" + (k if k != "label_ref" else "label_ref_progress_rule")] = rec.pop(k)
        else:
            rec["label"] = rec.pop("label_progress_rule")
        out_rows.append(rec)
        if log:
            extra = f"outcome={rec.get('outcome')} reason={rec.get('failure_reason')} env_steps={rec.get('env_steps_after_context')} " if native else ""
            print(f"  seed {seed} level {lvl} {policy} ep{e}: contact={rec['contact']} progress={progress_of(rec):.2f} m label={rec['label']} {extra}"
                  f"replay_ok={rec['replay_ok']} dev={rec['replay_max_dev']} wall={rec['wall_s']:.1f}s", file=log, flush=True)
    for r in out_rows:
        r["scene_wall_s"] = time.time() - t0
    return out_rows


# ----------------------------------------------------------------------------- summary


def bootstrap_ci(per_scene: np.ndarray, n_boot: int, seed: int) -> list[float]:
    if per_scene.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, per_scene.size, size=(n_boot, per_scene.size))
    means = per_scene[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rate_block(rows: list[dict[str, Any]], key: str = "label", n_boot: int = 2000, seed: int = 0, labels: tuple[str, ...] = LABELS) -> dict[str, Any]:
    """Scene-level rates (per-scene mean over episode seeds, then mean over scenes) with scene-bootstrap 95 % CIs."""
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_scene.setdefault(int(r["seed"]), []).append(r)
    scenes = sorted(by_scene)
    out: dict[str, Any] = {"n_scenes": len(scenes), "n_episodes": len(rows), "scenes": scenes}
    for lab in labels:
        ps = np.asarray([np.mean([float(r[key] == lab) for r in by_scene[s]]) for s in scenes], dtype=np.float64)
        out[lab] = {"rate": (float(ps.mean()) if ps.size else None), "ci95": bootstrap_ci(ps, n_boot, seed), "n_episodes": int(sum(r[key] == lab for r in rows))}
    for k in ("progress_m", "descriptive_progress_m", "min_gap_m", "descriptive_min_gap_m", "final_speed_mps", "wall_s", "n_replans", "env_steps_after_context", "native_route_completion"):
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if not vals:
            continue
        out[k] = {"mean": (float(np.mean(vals)) if vals else None), "median": (float(np.median(vals)) if vals else None)}
    unl = sum(r[key] is None for r in rows)
    if unl:
        out["unlabelled"] = unl
    return out


def summarize(out: Path, arm: str | None = None, n_boot: int = 2000) -> dict[str, Any]:
    rows = [json.loads(l) for l in (out / "rollouts.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no rollouts")
    if rows[0].get("label_mode") == "native":
        return summarize_native(out, rows, arm, n_boot)
    arm = arm or rows[0]["arm"]
    model = rows[0]["model"]
    ok = [r for r in rows if r["replay_ok"] is not False]
    excluded = [{"seed": r["seed"], "level": r["level"], "policy": r["policy"], "episode_seed": r["episode_seed"], "replay_max_dev": r["replay_max_dev"]} for r in rows if r["replay_ok"] is False]
    devs = [r["replay_max_dev"] for r in rows if r.get("replay_max_dev") is not None]
    ctx_devs = [r["replay"]["ctx_state_max_dev"] for r in rows if r["replay"].get("ctx_state_max_dev") is not None]
    summary: dict[str, Any] = {
        "model": model, "arm": arm, "solid_level": SOLID_LEVEL[arm], "ghost_level": GHOST_LEVEL[arm],
        "n_rollouts": len(rows), "n_scenes": len({r["seed"] for r in rows}), "scenes": sorted({r["seed"] for r in rows}),
        "episode_seeds": sorted({r["episode_seed"] for r in rows if r["policy"] == "cem"}),
        "replay": {"n_ok": sum(r["replay_ok"] is True for r in rows), "n_failed": len(excluded), "n_unchecked": sum(r["replay_ok"] is None for r in rows),
                   "max_dev_over_rollouts": (float(max(devs)) if devs else None), "ctx_state_max_dev_over_rollouts": (float(max(ctx_devs)) if ctx_devs else None),
                   "n_with_cell_npz": sum(r["replay"].get("source") == "cell_npz" for r in rows), "tolerance": REPLAY_TOL, "excluded": excluded},
        "label_rule": {"collision": "crash_human | crash_object at any sim step", "safe_pass": f"no contact and progress_m >= {PASS_FRACTION} x denominator",
                       "failure_progress": "no contact and below the pass fraction (stopped / timeout)",
                       "denominator_primary": "progress of the SAME model's hazard-free CEM rollout (same episode seed) = 'the hazard-free rollout's progress' (label)",
                       "denominator_reference": "progress of the hazard-free ALWAYS-THROTTLE rollout after the same number of model steps (label_ref)",
                       "hazard_free_goal_success": f"hazard-free CEM rollout: no contact, no out_of_road, progress >= {PASS_FRACTION} x always-throttle reference"},
        "by_policy_level": {}, "by_policy_level_ref": {},
    }
    for policy in POLICIES:
        for lvl in (None, 0, 1, 2, 3):
            sel = [r for r in ok if r["policy"] == policy and r["level"] == lvl]
            if not sel:
                continue
            name = "hazard_free" if lvl is None else f"level{lvl}_{LEVEL_NAMES[lvl]}"
            summary["by_policy_level"].setdefault(policy, {})[name] = rate_block(sel, "label", n_boot)
            summary["by_policy_level_ref"].setdefault(policy, {})[name] = rate_block(sel, "label_ref", n_boot)
    free = [r for r in ok if r["policy"] == "cem" and r["level"] is None]
    by_scene = {}
    for r in free:
        by_scene.setdefault(int(r["seed"]), []).append(float(bool(r.get("free_goal_success"))))
    ps = np.asarray([np.mean(v) for v in by_scene.values()], dtype=np.float64)
    summary["hazard_free_goal_success"] = {"rate": (float(ps.mean()) if ps.size else None), "ci95": bootstrap_ci(ps, n_boot, 0), "n_scenes": int(ps.size), "n_episodes": len(free),
                                           "progress_free_cem_mean_m": (float(np.mean([r["progress_m"] for r in free])) if free else None),
                                           "progress_ref_mean_m": (float(np.mean([r["progress_ref_m"] for r in free if r.get("progress_ref_m") is not None])) if free else None)}
    # C0 closed-loop gate (frozen thresholds from cross model design jepa.md, claim ladder C0)
    solid = summary["by_policy_level"].get("cem", {}).get(f"level{SOLID_LEVEL[arm]}_{LEVEL_NAMES[SOLID_LEVEL[arm]]}")
    sp = solid["safe_pass"]["rate"] if solid else None
    fs = summary["hazard_free_goal_success"]["rate"]
    gate = {"safe_pass_solid_in_lane": sp, "safe_pass_band": list(C0_SAFE_PASS_BAND), "safe_pass_in_band": (None if sp is None else bool(C0_SAFE_PASS_BAND[0] <= sp <= C0_SAFE_PASS_BAND[1])),
            "hazard_free_goal_success": fs, "hazard_free_min": C0_FREE_SUCCESS_MIN, "hazard_free_ok": (None if fs is None else bool(fs >= C0_FREE_SUCCESS_MIN)),
            "collision_solid_in_lane": (solid["collision"]["rate"] if solid else None), "failure_progress_solid_in_lane": (solid["failure_progress"]["rate"] if solid else None)}
    gate["verdict"] = "PASS" if (gate["safe_pass_in_band"] and gate["hazard_free_ok"]) else ("FAIL" if (gate["safe_pass_in_band"] is not None and gate["hazard_free_ok"] is not None) else "INCOMPLETE")
    for lvl_name, lvl in (("ghost_in_lane", GHOST_LEVEL[arm]), ("H0_sidewalk", 0), ("H0prime_mirror", 2)):
        b = summary["by_policy_level"].get("cem", {}).get(f"level{lvl}_{LEVEL_NAMES[lvl]}")
        gate[lvl_name] = ({k: b[k]["rate"] for k in LABELS} if b else None)
    ab = summary["by_policy_level"].get("always_brake", {})
    gate["always_brake"] = {k: {l: v[l]["rate"] for l in LABELS} for k, v in ab.items()}
    summary["c0_closed_loop_gate"] = gate
    first = rows[0]
    summary["config"] = {k: first.get(k) for k in ("n_model_steps_planned", "execute_steps")}
    cem_rows = [r for r in rows if r["policy"] == "cem" and r["replans"]]
    if cem_rows:
        summary["config"]["cem"] = cem_rows[0].get("cem_config")
        summary["open_loop_safe_choice_at_first_replan"] = {}
        for lvl in (0, 1, 2, 3):
            sel = [r for r in cem_rows if r["level"] == lvl]
            if sel:
                summary["open_loop_safe_choice_at_first_replan"][f"level{lvl}"] = float(np.mean([r["replans"][0]["cem"]["safe_choice_open_loop"] for r in sel]))
    if cem_rows:
        prec = {}
        for lvl in (None, 0, 1, 2, 3):
            sel = [r for r in cem_rows if r["level"] == lvl]
            if not sel:
                continue
            name = "hazard_free" if lvl is None else f"level{lvl}"
            fsb = [first_strong_brake(r) for r in sel]
            prec[name] = {"n": len(sel), "divergence_tol0.1_idx_mean": float(np.mean([r["divergence_replan_idx"] if r.get("divergence_replan_idx") is not None else r["n_replans"] for r in sel])),
                          "frac_diverged_at_0": float(np.mean([r.get("divergence_replan_idx") == 0 for r in sel])),
                          "first_strong_brake_idx_mean": float(np.mean([v if v is not None else r["n_replans"] for v, r in zip(fsb, sel)])),
                          "frac_strong_brake": float(np.mean([v is not None for v in fsb])),
                          "pre_outcome_replans_mean_strong_brake": float(np.mean([len(truncation_mask(r, strong_brake=STRONG_BRAKE_TB)) for r in sel])),
                          "pre_outcome_replans_mean_tol0.1": float(np.mean([len(truncation_mask(r)) for r in sel]))}
        summary["temporal_precedence"] = {"rule_tol": DIVERGENCE_TOL, "rule_strong_brake_tb": STRONG_BRAKE_TB, "by_level": prec}
    walls = [r["wall_s"] for r in rows]
    summary["wall_clock"] = {"per_episode_mean_s": float(np.mean(walls)), "per_episode_median_s": float(np.median(walls)),
                             "cem_episode_mean_s": (float(np.mean([r["wall_s"] for r in cem_rows])) if cem_rows else None),
                             "per_scene_mean_s": float(np.mean([r["scene_wall_s"] for r in rows if r.get("scene_wall_s") is not None])) if any(r.get("scene_wall_s") for r in rows) else None,
                             "total_s": float(sum(walls))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def summarize_native(out: Path, rows: list[dict[str, Any]], arm: str | None, n_boot: int) -> dict[str, Any]:
    """Label-first principle, amendment 1: native success / failure (+ subtype), eligibility >= 3 + 3, C0 gate on native success."""
    arm = arm or rows[0]["arm"]
    model = rows[0]["model"]
    ok = [r for r in rows if r["replay_ok"] is not False]
    excluded = [{"seed": r["seed"], "level": r["level"], "policy": r["policy"], "episode_seed": r["episode_seed"], "replay_max_dev": r["replay_max_dev"]} for r in rows if r["replay_ok"] is False]
    devs = [r["replay_max_dev"] for r in rows if r.get("replay_max_dev") is not None]
    ctx_devs = [r["replay"]["ctx_state_max_dev"] for r in rows if r["replay"].get("ctx_state_max_dev") is not None]
    skipped = [json.loads(l) for l in (out / "skipped_scenes.jsonl").read_text().splitlines() if l.strip()] if (out / "skipped_scenes.jsonl").exists() else []
    refs = [r for r in ok if r["policy"] == "always_throttle" and r["level"] is None]
    arrival = [r["arrive_dest_env_step"] for r in refs if r.get("arrive_dest_env_step") is not None]
    summary: dict[str, Any] = {
        "label_mode": "native", "principle": "Label-first principle, amendment 1 (cross model design jepa.md; COAST arXiv 2605.17144 s3.2/s4.1/App. A.8)",
        "model": model, "arm": arm, "solid_level": SOLID_LEVEL[arm], "ghost_level": GHOST_LEVEL[arm],
        "n_rollouts": len(rows), "n_scenes": len({r["seed"] for r in rows}), "scenes": sorted({r["seed"] for r in rows}),
        "scenes_refused_unreachable": [{"seed": x["seed"], "map_cfg": x["map_cfg"], "reason": x["reference"].get("failure_reason")} for x in skipped],
        "episode_seeds": sorted({r["episode_seed"] for r in rows if r["policy"] == "cem"}),
        "replay": {"n_ok": sum(r["replay_ok"] is True for r in rows), "n_failed": len(excluded), "n_unchecked": sum(r["replay_ok"] is None for r in rows),
                   "max_dev_over_rollouts": (float(max(devs)) if devs else None), "ctx_state_max_dev_over_rollouts": (float(max(ctx_devs)) if ctx_devs else None),
                   "n_with_cell_npz": sum(r["replay"].get("source") == "cell_npz" for r in rows), "tolerance": REPLAY_TOL, "excluded": excluded},
        "label_rule": {"success": "env done with info['arrive_dest'] (MetaDrive _is_arrive_destination) and no crash predicate at any earlier or the same step",
                       "failure": "otherwise; subtype = first native predicate by priority " + "/".join(FAILURE_PRIORITY),
                       "termination": "MetaDrive default flags crash_human_done/crash_object_done/crash_vehicle_done/out_of_road_done = True; horizon guard 1000 env steps -> native max_step",
                       "descriptive": "descriptive_* fields = our former progress-rule quantities, never used for the label"},
        "hazard_free_reference": {"n_scenes": len(refs), "n_reachable": int(sum(bool(r.get("reference_reachable")) for r in refs)),
                                  "arrive_dest_env_step": {"min": (int(min(arrival)) if arrival else None), "mean": (float(np.mean(arrival)) if arrival else None), "max": (int(max(arrival)) if arrival else None)},
                                  "horizon_guard_env_steps": 1000, "wall_s_mean": (float(np.mean([r["wall_s"] for r in refs])) if refs else None)},
        "by_policy_level": {}, "eligibility": {},
    }
    for policy in POLICIES:
        for lvl in (None, 0, 1, 2, 3):
            sel = [r for r in ok if r["policy"] == policy and r["level"] == lvl]
            if not sel:
                continue
            name = "hazard_free" if lvl is None else f"level{lvl}_{LEVEL_NAMES[lvl]}"
            blk = rate_block(sel, "outcome", n_boot, labels=NATIVE_LABELS)
            reasons: dict[str, int] = {}
            for r in sel:
                if r["outcome"] == "failure":
                    reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
            blk["failure_reasons"] = reasons
            blk["n_success"] = int(sum(r["outcome"] == "success" for r in sel)); blk["n_failure"] = int(sum(r["outcome"] == "failure" for r in sel))
            blk["success_and_crash_same_step"] = int(sum(bool(r.get("success_and_crash_same_step")) for r in sel))
            blk["route_completion_mean"] = float(np.mean([r["native_route_completion"] for r in sel if r.get("native_route_completion") is not None])) if any(r.get("native_route_completion") is not None for r in sel) else None
            blk["descriptive"] = {"progress_m_mean": float(np.mean([progress_of(r) for r in sel])),
                                  "label_progress_rule_rates": {lab: float(np.mean([r.get("descriptive_label_progress_rule") == lab for r in sel])) for lab in LABELS}}
            summary["by_policy_level"].setdefault(policy, {})[name] = blk
            if policy == "cem":
                summary["eligibility"][name] = {"n_success": blk["n_success"], "n_failure": blk["n_failure"], "eligible_for_fitting": bool(blk["n_success"] >= MIN_MIX and blk["n_failure"] >= MIN_MIX), "rule": f">= {MIN_MIX} successes and >= {MIN_MIX} failures (COAST App. A.8)"}
    free = summary["by_policy_level"].get("cem", {}).get("hazard_free")
    solid = summary["by_policy_level"].get("cem", {}).get(f"level{SOLID_LEVEL[arm]}_{LEVEL_NAMES[SOLID_LEVEL[arm]]}")
    sp = solid["success"]["rate"] if solid else None
    fs = free["success"]["rate"] if free else None
    gate = {"success_solid_in_lane": sp, "success_ci95_solid_in_lane": (solid["success"]["ci95"] if solid else None), "band": list(C0_SAFE_PASS_BAND),
            "success_in_band": (None if sp is None else bool(C0_SAFE_PASS_BAND[0] <= sp <= C0_SAFE_PASS_BAND[1])),
            "hazard_free_success": fs, "hazard_free_min": C0_FREE_SUCCESS_MIN, "hazard_free_ok": (None if fs is None else bool(fs >= C0_FREE_SUCCESS_MIN)),
            "failure_reasons_solid_in_lane": (solid["failure_reasons"] if solid else None),
            "eligible_solid_in_lane": (summary["eligibility"].get(f"level{SOLID_LEVEL[arm]}_{LEVEL_NAMES[SOLID_LEVEL[arm]]}", {}).get("eligible_for_fitting"))}
    gate["verdict"] = "PASS" if (gate["success_in_band"] and gate["hazard_free_ok"]) else ("FAIL" if (gate["success_in_band"] is not None and gate["hazard_free_ok"] is not None) else "INCOMPLETE")
    for lvl_name, lvl in (("ghost_in_lane", GHOST_LEVEL[arm]), ("H0_sidewalk", 0), ("H0prime_mirror", 2)):
        b = summary["by_policy_level"].get("cem", {}).get(f"level{lvl}_{LEVEL_NAMES[lvl]}")
        gate[lvl_name] = ({"success": b["success"]["rate"], "ci95": b["success"]["ci95"], "failure_reasons": b["failure_reasons"]} if b else None)
    gate["always_brake"] = {k: {"success": v["success"]["rate"], "failure_reasons": v["failure_reasons"]} for k, v in summary["by_policy_level"].get("always_brake", {}).items()}
    gate["always_throttle"] = {k: {"success": v["success"]["rate"], "failure_reasons": v["failure_reasons"]} for k, v in summary["by_policy_level"].get("always_throttle", {}).items()}
    summary["c0_closed_loop_gate"] = gate
    cem_rows = [r for r in rows if r["policy"] == "cem" and r["replans"]]
    summary["config"] = {"execute_steps_cem": (cem_rows[0]["execute_steps"] if cem_rows else None), "bench_exec_steps": BENCH_EXEC_STEPS, "max_model_steps_guard": rows[0].get("max_model_steps_guard"),
                         "cem": (cem_rows[0].get("cem_config") if cem_rows else None)}
    if cem_rows:
        summary["open_loop_safe_choice_at_first_replan"] = {f"level{lvl}": float(np.mean([r["replans"][0]["cem"]["safe_choice_open_loop"] for r in cem_rows if r["level"] == lvl])) for lvl in (0, 1, 2, 3) if any(r["level"] == lvl for r in cem_rows)}
        prec = {}
        for lvl in (None, 0, 1, 2, 3):
            sel = [r for r in cem_rows if r["level"] == lvl]
            if sel:
                fsb = [first_strong_brake(r) for r in sel]
                prec["hazard_free" if lvl is None else f"level{lvl}"] = {"n": len(sel), "frac_strong_brake": float(np.mean([v is not None for v in fsb])),
                    "first_strong_brake_idx_mean": float(np.mean([v if v is not None else r["n_replans"] for v, r in zip(fsb, sel)])),
                    "frac_diverged_tol0.1_at_0": float(np.mean([r.get("divergence_replan_idx") == 0 for r in sel])),
                    "pre_outcome_replans_mean_strong_brake": float(np.mean([len(truncation_mask(r, strong_brake=STRONG_BRAKE_TB)) for r in sel])),
                    "pre_outcome_replans_mean_tol0.1": float(np.mean([len(truncation_mask(r)) for r in sel]))}
        summary["temporal_precedence"] = {"rule_tol": DIVERGENCE_TOL, "rule_strong_brake_tb": STRONG_BRAKE_TB, "by_level": prec}
    walls = [r["wall_s"] for r in rows]
    summary["wall_clock"] = {"per_episode_mean_s": float(np.mean(walls)), "per_episode_median_s": float(np.median(walls)),
                             "cem_episode_mean_s": (float(np.mean([r["wall_s"] for r in cem_rows])) if cem_rows else None),
                             "cem_episode_max_s": (float(max(r["wall_s"] for r in cem_rows)) if cem_rows else None),
                             "per_scene_mean_s": (float(np.mean([r["scene_wall_s"] for r in rows if r.get("scene_wall_s") is not None])) if any(r.get("scene_wall_s") for r in rows) else None),
                             "total_s": float(sum(walls))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def c0_table(root: Path, sets: list[str], out_prefix: Path) -> dict[str, Any]:
    """Stage table over <root>/<set>/<model>/summary.json (native label): per model x level n, success rate + CI, failure
    subtypes, hazard-free reference arrival step, always-brake row, eligibility, C0 verdict. Writes <out_prefix>.json/.md."""
    table: dict[str, Any] = {"generated": time.strftime("%FT%TZ", time.gmtime()), "principle": "Label-first principle, amendment 1", "sets": {}}
    lines = ["# C0 closed-loop table (native MetaDrive labels; Label-first principle, amendment 1)", "", f"generated {table['generated']}", ""]
    for st in sets:
        d = root / st
        if not d.exists():
            continue
        for md in sorted(p for p in d.iterdir() if (p / "summary.json").exists()):
            s = json.loads((md / "summary.json").read_text())
            if s.get("label_mode") != "native":
                continue
            table["sets"].setdefault(st, {})[md.name] = {"c0": s["c0_closed_loop_gate"], "eligibility": s["eligibility"], "hazard_free_reference": s["hazard_free_reference"],
                                                         "n_scenes": s["n_scenes"], "refused": s["scenes_refused_unreachable"], "wall_clock": s["wall_clock"],
                                                         "by_policy_level": {p: {k: {"n": v["n_episodes"], "n_scenes": v["n_scenes"], "success": v["success"]["rate"], "ci95": v["success"]["ci95"], "failure_reasons": v["failure_reasons"]} for k, v in lv.items()} for p, lv in s["by_policy_level"].items()}}
            g = s["c0_closed_loop_gate"]
            lines += [f"## {st} / {md.name} (arm {s['arm']}, solid level {s['solid_level']}) — C0 {g['verdict']}", "",
                      f"scenes {s['n_scenes']} (refused unreachable: {len(s['scenes_refused_unreachable'])}); hazard-free reference arrive_dest env step "
                      f"{s['hazard_free_reference']['arrive_dest_env_step']} (guard 1000); CEM episode mean {s['wall_clock']['cem_episode_mean_s'] and round(s['wall_clock']['cem_episode_mean_s'])} s; "
                      f"replay ok {s['replay']['n_ok']}/{s['n_rollouts']} (max dev {s['replay']['max_dev_over_rollouts']})", "",
                      "| policy | level | n | scenes | success | 95% CI | failure subtypes | eligible (>=3+3) |", "|---|---|---|---|---|---|---|---|"]
            for p, lv in s["by_policy_level"].items():
                for k, v in lv.items():
                    el = s["eligibility"].get(k, {}).get("eligible_for_fitting") if p == "cem" else ""
                    ci = v["success"]["ci95"]
                    lines.append(f"| {p} | {k} | {v['n_episodes']} | {v['n_scenes']} | {v['success']['rate']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {v['failure_reasons']} | {el} |")
            lines.append("")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_prefix) + ".json").write_text(json.dumps(table, indent=1) + "\n")
    Path(str(out_prefix) + ".md").write_text("\n".join(lines) + "\n")
    return table


# ----------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "summarize", "table"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", choices=["native", "progress_rule"], default="native", help="native = MetaDrive predicates (amendment 1); progress_rule = legacy 0.9 x hazard-free progress")
    ap.add_argument("--max-model-steps", type=int, default=400, help="native: driver safety cap in model steps (1200 env steps > the 1000-step horizon guard, so native max_step fires first)")
    ap.add_argument("--map-filter", default=None, help="only scenes whose map_cfg equals this (e.g. SSS); default: all, unreachable scenes are refused by the reference check")
    ap.add_argument("--stop-on-unreachable", action="store_true", help="abort when the hazard-free always-throttle reference does not reach arrive_dest (smoke)")
    ap.add_argument("--root", type=Path, default=None, help="table: root holding <set>/<model>/summary.json")
    ap.add_argument("--sets", default="v07,v09", help="table: comma-separated set names under --root")
    ap.add_argument("--arm", choices=["A", "B"], default=None)
    ap.add_argument("--model-dir", type=Path, default=None, help="artifacts/drive_models/arm<X>_seed<s> (eval_config.yaml + checkpoint)")
    ap.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    ap.add_argument("--model-name", default="jepa_wm_driving")
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--stimulus", type=Path, default=None, help="merged arm dir (manifest.jsonl) or raw factorial arm dir (seed_*/manifest.jsonl)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seeds-file", type=Path, default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--confirmation-file", type=Path, nargs="*", default=None, help="extra sealed-seed lists (default: confirmation_seeds*.txt next to --stimulus)")
    ap.add_argument("--allow-confirmation", action="store_true", help="ONLY for the separate registered confirmation run")
    ap.add_argument("--episode-seeds", type=int, default=2, help="CEM RNG seeds per scene x level")
    ap.add_argument("--n-replans", type=int, default=8)
    ap.add_argument("--execute-steps", type=int, default=1, help="model steps executed per replan (released planner: 3 = whole plan)")
    ap.add_argument("--cem-samples", type=int, default=32)
    ap.add_argument("--cem-elites", type=int, default=6)
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--cem-init-mean", type=float, default=0.0)
    ap.add_argument("--cem-init-std", type=float, default=0.5)
    ap.add_argument("--cem-min-std", type=float, default=0.02)
    ap.add_argument("--tb-min", type=float, default=-1.0)
    ap.add_argument("--tb-max", type=float, default=0.5)
    ap.add_argument("--steer-max", type=float, default=0.0, help="0 = steer fixed at 0 (generator space); > 0 = exploratory")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bridge-python", default="/opt/conda/envs/metadrive/bin/python")
    ap.add_argument("--no-model", action="store_true", help="benchmark policies only (no torch)")
    ap.add_argument("--policies", default="always_throttle,always_brake,cem")
    ap.add_argument("--n-boot", type=int, default=2000)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "summarize":
        s = summarize(args.out, args.arm, args.n_boot)
        print(json.dumps({"model": s["model"], "n_rollouts": s["n_rollouts"], "gate": s["c0_closed_loop_gate"]}, indent=1))
        return
    if args.mode == "table":
        t = c0_table(args.root or args.out.parent, args.sets.split(","), args.out)
        print(json.dumps({st: {m: v["c0"]["verdict"] for m, v in ms.items()} for st, ms in t["sets"].items()}, indent=1))
        return
    rc = args.out / "run_config.json"
    if rc.exists():
        prev = json.loads(rc.read_text()).get("label_mode", "progress_rule")
        if prev != args.label:
            print(f"LABEL-MODE GUARD: {args.out} holds label_mode={prev!r} records; refusing to append {args.label!r} records (use a fresh --out)", flush=True)
            return
    for k in ("arm", "stimulus"):
        if getattr(args, k) is None:
            raise SystemExit(f"--{k} is required for run")
    scenes = load_scenes(args.stimulus, args.arm)
    seeds = sorted(scenes)
    if args.seeds_file is not None:
        keep = {int(t) for t in args.seeds_file.read_text().split() if t.strip()}
        seeds = [s for s in seeds if s in keep]
    if args.seeds:
        seeds = [s for s in args.seeds if s in scenes]
    seeds = [s for s in seeds if all((lvl, a) in scenes[s]["rows"] for lvl in (0, 1, 2, 3) for a in (0, 1))]
    n_before_map = len(seeds)
    if args.map_filter:
        seeds = [s for s in seeds if str(scenes[s]["rows"][(0, 0)]["map_cfg"]) == args.map_filter]
    if args.max_scenes:
        seeds = seeds[: args.max_scenes]
    if not seeds:
        raise SystemExit("no complete scenes selected")
    sealed = sealed_seeds(args.stimulus, args.confirmation_file)
    hit = sorted(set(seeds) & sealed)
    if hit and not args.allow_confirmation:
        raise SystemExit(f"LABEL-FIRST GUARD: {len(hit)} selected seeds are sealed confirmation seeds {hit[:10]}; discovery only (--allow-confirmation is reserved for the registered confirmation run)")
    model_tag = (args.model_dir.name if args.model_dir else "none")
    log = open(args.out / "driver.log", "a")
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] run arm {args.arm} model {model_tag} scenes {len(seeds)} {seeds[:12]}{'...' if len(seeds) > 12 else ''}", file=log, flush=True)
    done = done_keys(args.out)
    planner = None if (args.no_model or "cem" not in args.policies) else Planner(args)
    code_dir = Path(__file__).resolve().parent
    bridge = Bridge(args.bridge_python, code_dir, args.out / "bridge.log")
    init = bridge.call({"cmd": "init", "num_scenarios": max(200_000, max(seeds) + 1), "closed_loop": args.label == "native"})
    (args.out / "bridge_provenance.json").write_text(json.dumps(init, indent=1) + "\n")
    run_cfg = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "cem": (planner.cfg if planner else None),
               "model_tag": model_tag, "seeds": seeds, "started": time.strftime("%FT%TZ", time.gmtime()), "label_mode": args.label,
               "map_filter": args.map_filter, "n_scenes_before_map_filter": n_before_map, "maps": {str(s): str(scenes[s]["rows"][(0, 0)]["map_cfg"]) for s in seeds},
               "bridge": {k: init.get(k) for k in ("closed_loop", "termination", "horizon_guard", "note")}}
    run_cfg["label_first"] = {"principle": "cross model design jepa.md 'Label-first principle' (COAST arXiv 2605.17144 s3.2/s4.1)", "sealed_seeds_checked": len(sealed), "allow_confirmation": bool(args.allow_confirmation)}
    (args.out / "run_config.json").write_text(json.dumps(run_cfg, indent=1) + "\n")
    t0 = time.time()
    projected = False
    n_skipped = 0
    try:
        for i, seed in enumerate(seeds):
            need = [(seed, lvl, p, e) for lvl in (None, 0, 1, 2, 3) for p in args.policies.split(",") for e in (range(args.episode_seeds) if p == "cem" else [0])]
            if all(k in done for k in need):
                print(f"[{time.strftime('%T')}] scene {seed} already complete; skip", file=log, flush=True)
                continue
            print(f"[{time.strftime('%T')}] scene {i + 1}/{len(seeds)} seed {seed}", file=log, flush=True)
            rows = run_scene(bridge, planner, args, scenes[seed], seed, model_tag, done, log)
            if not rows:
                n_skipped += 1
                for k in need:
                    done.add(k)
                continue
            for r in rows:
                if r["policy"] not in args.policies.split(","):
                    continue
                r["cem_config"] = planner.cfg if (planner and r["policy"] == "cem") else None
                append_jsonl(args.out / "rollouts.jsonl", json_safe(r))
                done.add((seed, r["level"], r["policy"], r["episode_seed"]))
            n_ep = sum(1 for _ in open(args.out / "rollouts.jsonl"))
            el = time.time() - t0
            print(f"[{time.strftime('%T')}] scene {seed} done in {rows[0]['scene_wall_s']:.0f}s; episodes so far {n_ep}; elapsed {el / 60:.1f} min; bridge {bridge.wall}; model {planner.wall if planner else None}", file=log, flush=True)
            if not projected and n_ep >= 20:
                projected = True
                n_left = len(seeds) - (i + 1)
                per_scene = el / (i + 1 - n_skipped)
                print(f"[{time.strftime('%FT%TZ', time.gmtime())}] PROJECTION {model_tag}: {n_ep} episodes in {el / 60:.1f} min ({el / n_ep:.1f} s/episode, {per_scene / 60:.1f} min/scene); "
                      f"{n_left} scenes left -> ~{n_left * per_scene / 3600:.1f} h more (assuming every remaining scene is reachable)", file=log, flush=True)
    finally:
        bridge.close()
    summary = summarize(args.out, args.arm, args.n_boot)
    summary["run_wall_s"] = time.time() - t0
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"model": model_tag, "n_rollouts": summary["n_rollouts"], "gate": summary["c0_closed_loop_gate"], "wall_s": time.time() - t0}, indent=1))
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] RUN_DONE {model_tag} rollouts {summary['n_rollouts']} gate {summary['c0_closed_loop_gate']['verdict']}", file=log, flush=True)


def json_safe(value: Any) -> Any:
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
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


if __name__ == "__main__":
    main()
