#!/usr/bin/env python3
"""Hazard-free-only planner diagnostics for the closed-loop adapter (coordinator instruction 2026-09-03 ~17:55 UTC).

Why: under the registered rolling-goal progress currency the compact CEM STALLS in the hazard-free scene (native max_step at
995 env steps, 16.6 m) while always-throttle reaches arrive_dest at env step 128. With zero hazard-free successes the native
label cannot be fitted, so - as COAST chooses a checkpoint on success rates - the planner configuration is diagnosed on
HAZARD-FREE scenes only (no hazard cell is rolled out; nothing about hazard labels is inspected).

Configs (all hazard-free, native MetaDrive termination via closed_loop_bridge.py, steer fixed at 0, throttle_brake in [-1, 0.5]):
  compact_rolling3   our compact CEM (32/6/3, init N(0, 0.5)), rolling goal = hazard-free always-throttle frame 3 model steps
                     ahead of the executed count (the registered currency), execute 1 model step. Full cost traces per replan:
                     cost of the brake / throttle / final candidates, the ENCODER-ONLY distance of the current frame to the goal
                     (cost of standing still without any prediction), and the encoder-only distance of the reference frames at
                     executed .. executed+3 to that goal (does the currency have a gradient along the true throttle path?).
  compact_dest       same CEM, goal IMAGE = the reference frame at arrive_dest (goal-image planning as in the jepa-wms evals).
  compact_rolling6 / compact_rolling10   receding goal 6 / 10 model steps ahead.
  released_cem_dest  the released jepa-wms CEMPlanner (configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_*.yaml):
                     iterations 15, num_samples 300, num_elites 10, horizon 3, var_scale 0.1 (init std), momentum 0, mean-sample
                     inclusion, clipping, num_act_stepped 3 (execute the whole plan), L2 objective on the LAST predicted frame
                     over ALL tokens (their ReprTargetDistMPCObjective, sum_all_diffs False), goal image = destination frame.
Each episode: native outcome (arrive_dest / max_step / driver cap) or ``diag_timeout`` when --max-wall-per-episode is hit
(reported as such, never as a native outcome), arrival env step, progress, route_completion, wall-clock.
Output: <out>/diag.jsonl (one line per episode incl. the per-replan trace) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_rollout as clr  # noqa: E402
from protocol import append_jsonl  # noqa: E402

CONFIGS = {
    "compact_rolling3": {"kind": "compact", "goal": "rolling", "ahead": 3, "execute": 1, "samples": 32, "elites": 6, "iters": 3, "init_std": 0.5, "tokens": "hazard_corridor"},
    "compact_dest": {"kind": "compact", "goal": "dest", "ahead": None, "execute": 1, "samples": 32, "elites": 6, "iters": 3, "init_std": 0.5, "tokens": "hazard_corridor"},
    "compact_rolling6": {"kind": "compact", "goal": "rolling", "ahead": 6, "execute": 1, "samples": 32, "elites": 6, "iters": 3, "init_std": 0.5, "tokens": "hazard_corridor"},
    "compact_rolling10": {"kind": "compact", "goal": "rolling", "ahead": 10, "execute": 1, "samples": 32, "elites": 6, "iters": 3, "init_std": 0.5, "tokens": "hazard_corridor"},
    # range-k diagnostic (coordinator, 18:35 UTC): FIXED goal image = reference frame k model steps ahead of the context; the
    # episode ends (descriptively) when the ego passes the goal frame's longitudinal position, or at the env's own done.
    **{f"fixed_k{k}": {"kind": "compact", "goal": "fixed", "ahead": k, "execute": 1, "samples": 32, "elites": 6, "iters": 3, "init_std": 0.5, "tokens": "hazard_corridor"} for k in (6, 10, 20, 40)},
    "released_k10": {"kind": "released", "goal": "fixed", "ahead": 10, "execute": 3, "samples": 300, "elites": 10, "iters": 15, "init_std": 0.1, "tokens": "all"},
    "released_cem_dest": {"kind": "released", "goal": "dest", "ahead": None, "execute": 3, "samples": 300, "elites": 10, "iters": 15, "init_std": 0.1, "tokens": "all",
                          "source": "vendor/jepa-wms configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml (planner: cem, iterations 15, num_samples 300, num_elites 10, horizon 3, var_scale 0.1, momentum 0, num_act_stepped 3, objective L2)"},
}


def released_cem_plan(planner: clr.Planner, z_ctx, z_goal, idx: np.ndarray, rng: np.random.RandomState, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """The released CEMPlanner loop in our action space: mean 0, std var_scale, N samples with the mean as sample 0, clip,
    top-k elites -> new mean/std (momentum 0), returns the final mean. Costs = L2 over ``idx`` tokens of the last frame."""
    H = clr.HORIZON
    mean = np.zeros(H); std = np.full(H, cfg["init_std"])
    hist = []
    brake = np.repeat(np.asarray([clr.BRAKE_CHUNK]), H, axis=0); throttle = np.repeat(np.asarray([clr.THROTTLE_CHUNK]), H, axis=0)
    for it in range(cfg["iters"]):
        tb = mean[None] + std[None] * rng.randn(cfg["samples"], H)
        tb[0] = mean  # mean-sample inclusion trick
        tb = np.clip(tb, planner.cfg["tb_min"], planner.cfg["tb_max"])
        cands = np.stack([np.zeros_like(tb), tb], axis=-1).astype(np.float32)
        batch = np.concatenate([cands, brake[None], throttle[None]], axis=0).astype(np.float32)
        cost = planner.costs(z_ctx, batch, z_goal, idx)
        cs = cost[: cfg["samples"]]
        elite = np.argsort(cs)[: cfg["elites"]]
        mean = tb[elite].mean(axis=0); std = tb[elite].std(axis=0)
        hist.append({"iter": it, "cost_min": float(cs.min()), "cost_elite_mean": float(cs[elite].mean()), "cost_brake": float(cost[-2]), "cost_throttle": float(cost[-1]),
                     "mean_tb": mean.round(4).tolist(), "std_tb": std.round(4).tolist()})
    final = np.stack([np.zeros(H), np.clip(mean, planner.cfg["tb_min"], planner.cfg["tb_max"])], axis=-1)
    c_final = float(planner.costs(z_ctx, final[None].astype(np.float32), z_goal, idx)[0])
    return final, {"n_tokens": int(idx.size), "cost_final": c_final, "cost_brake": hist[-1]["cost_brake"], "cost_throttle": hist[-1]["cost_throttle"], "iters": hist}


def reference_rollout(bridge: clr.Bridge, args: argparse.Namespace, scene: dict[str, Any], seed: int) -> tuple[dict[str, Any], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    lat = {lvl: float(scene["rows"][(lvl, 0)]["hazard_lateral_m"]) for lvl in (0, 1, 2, 3)}
    rec = clr.run_one(bridge, None, args, scene, seed, None, "always_throttle", 0, None, args.max_model_steps, lat, args.model_tag, collect_frames=frames)
    return rec, frames


def run_config(name: str, cfg: dict[str, Any], bridge: clr.Bridge, planner: clr.Planner, args: argparse.Namespace, scene: dict[str, Any], seed: int,
               ref: dict[str, Any], frames: list[np.ndarray], episode_seed: int, log) -> dict[str, Any]:
    row0 = scene["rows"][(0, 0)]
    t_ep = time.time()
    resp = bridge.call({"cmd": "reset", "seed": seed, "map_cfg": str(row0["map_cfg"]), "row": row0, "v0": float(row0.get("v0_mps", 8.0)), "hazard": None})
    replay = clr.verify_replay(resp, row0, clr.cell_path(scene, row0), None)
    frame = clr.decode_frame(resp["frame"]); hmask, cmask = clr.decode_mask(resp["hazard_mask"]), clr.decode_mask(resp["corridor_mask"])
    rng = clr.episode_rng(args.model_tag + ":" + name, seed, None, episode_seed)
    dest_index = len(frames) - 1  # the reference's last frame = the frame at arrive_dest
    planner.cfg.update({"samples": cfg["samples"], "elites": cfg["elites"], "iters": cfg["iters"], "init_std": cfg["init_std"]})
    planner.clear_goal_cache()
    z_goal_cache: dict[int, Any] = {}
    executed = 0; steps_rec = []; trace = []; env_done = False; timeout = False; passed = False; pass_step = None
    ref_prog = ref.get("descriptive_progress_per_model_step_m") or ref.get("progress_per_model_step_m") or []
    target_progress = None
    if cfg["goal"] == "fixed":
        kk = min(cfg["ahead"], dest_index)
        target_progress = float(ref_prog[kk - 1]) if kk - 1 < len(ref_prog) else float(ref["descriptive_progress_m"])
    speed_before = float(resp["states"][clr.PREFIX_STEPS][6])
    n_replans = int(np.ceil(args.max_model_steps / cfg["execute"]))
    for k in range(n_replans):
        gi = dest_index if cfg["goal"] == "dest" else (min(cfg["ahead"], dest_index) if cfg["goal"] == "fixed" else min(executed + cfg["ahead"], dest_index))
        z_ctx = planner.encode(frame)
        if gi not in z_goal_cache:
            z_goal_cache[gi] = planner.goal_latent(f"{seed}:{gi}", frames[gi])
        z_goal = z_goal_cache[gi]
        idx = np.arange(z_goal.shape[0]) if cfg["tokens"] == "all" else planner.tokens(hmask, cmask)
        if idx.size == 0:
            idx = np.arange(z_goal.shape[0])
        if cfg["kind"] == "released":
            chunk, info = released_cem_plan(planner, z_ctx, z_goal, idx, rng, cfg)
        else:
            chunk, info = planner.plan(z_ctx, z_goal, idx, rng)
        rec: dict[str, Any] = {"replan": k, "model_steps_before": executed, "goal_index": gi, "n_tokens": int(idx.size),
                               "cost_final": info["cost_final"], "cost_brake": info["cost_brake"], "cost_throttle": info["cost_throttle"],
                               "margin_thr_vs_brake": float((info["cost_throttle"] - info["cost_brake"]) / max(info["cost_brake"], 1e-9)),
                               "mu_final_tb": [float(chunk[h, 1]) for h in range(chunk.shape[0])], "speed_before": speed_before,
                               "last_step_brake_artefact": bool(float(chunk[-1, 1]) <= -0.5 and speed_before > 0.5)}
        if name == "compact_rolling3" or args.full_trace:
            # encoder-only distances: standing still (current frame -> goal) and the reference frames executed..executed+ahead -> goal
            zc = z_ctx.reshape(-1, z_ctx.shape[-1]).float()
            ii = planner.torch.as_tensor(idx, device=planner.device)
            rec["cost_static_encoder_only"] = float(((zc[ii] - z_goal[ii]) ** 2).sum().item())
            prof = []
            for j in range(executed, min(gi, dest_index) + 1):
                zj = planner.goal_latent(f"{seed}:{j}", frames[j])
                prof.append(float(((zj[ii] - z_goal[ii]) ** 2).sum().item()))
            rec["ref_profile_encoder_only"] = prof  # reference frame at step executed .. gi vs goal gi (last entry = 0 when the goal frame itself)
            rec["cem_iters"] = info["iters"]
        n_exec = min(cfg["execute"], args.max_model_steps - executed)
        s = bridge.call({"cmd": "step", "chunk": np.asarray(chunk[:n_exec], dtype=np.float64).tolist(), "want_frames": True})
        n_done = int(np.ceil(s["n_sim_steps"] / clr.SIM_STEPS_PER_MODEL_STEP)); executed += n_done
        last = s["steps"][-1]
        rec.update({"executed_tb": [float(chunk[h, 1]) for h in range(n_done)], "speed_after": last["speed"], "progress_after_m": last["progress_m"], "route_completion": last.get("route_completion"),
                    "wall_s": time.time() - t_ep})
        speed_before = float(last["speed"])
        for j, x in enumerate(s["steps"]):
            steps_rec.append({"sim_step": executed * clr.SIM_STEPS_PER_MODEL_STEP - s["n_sim_steps"] + j + 1, **{kk: x[kk] for kk in x if kk != "state"}})
        trace.append(rec)
        if s["frames"]:
            frame = clr.decode_frame(s["frames"][-1]); hmask, cmask = clr.decode_mask(s["hazard_mask"][-1]), clr.decode_mask(s["corridor_mask"][-1])
        env_done = bool(s.get("done", False))
        if target_progress is not None and not passed and float(last["progress_m"]) >= target_progress:
            passed = True; pass_step = int(last.get("steps_after_context", 0))
            if not env_done:
                break  # descriptive pass of the goal frame's longitudinal position (range-k diagnostic)
        if env_done or executed >= args.max_model_steps:
            break
        if time.time() - t_ep > args.max_wall_per_episode:
            timeout = True
            break
    nat = clr.native_outcome(steps_rec, env_done, executed >= args.max_model_steps and not env_done)
    if passed and not env_done:
        nat["outcome"] = "descriptive_pass"; nat["failure_reason"] = None
    out = {"config": name, "config_spec": cfg, "model": args.model_tag, "arm": args.arm, "seed": seed, "map_cfg": str(row0["map_cfg"]), "episode_seed": episode_seed,
           "replay_ok": replay["ok"], "replay_max_dev": replay.get("max_dev"), "outcome": ("diag_timeout" if timeout else nat["outcome"]),
           "failure_reason": (None if timeout else nat["failure_reason"]), "native": nat, "arrive_dest_env_step": nat["arrive_dest_env_step"],
           "env_steps_after_context": nat["env_steps_after_context"], "progress_m": float(steps_rec[-1]["progress_m"]), "final_speed_mps": float(steps_rec[-1]["speed"]),
           "route_completion": steps_rec[-1].get("route_completion"), "n_replans": len(trace), "n_model_steps_executed": executed, "wall_s": time.time() - t_ep,
           "diag_timeout": timeout, "max_wall_per_episode_s": args.max_wall_per_episode, "reference_arrive_dest_env_step": ref["arrive_dest_env_step"],
           "reference_progress_m": ref["descriptive_progress_m"], "dest_goal_index": dest_index,
           "planner_cfg": dict(planner.cfg), "trace": trace,
           "range_k": (cfg["ahead"] if cfg["goal"] == "fixed" else None), "target_progress_m": target_progress, "passed_goal_position": passed, "pass_env_step": pass_step,
           "n_last_step_brake_artefact": int(sum(t["last_step_brake_artefact"] for t in trace)), "frac_last_step_brake_artefact": float(np.mean([t["last_step_brake_artefact"] for t in trace])),
           "margin_thr_vs_brake_mean": float(np.mean([t["margin_thr_vs_brake"] for t in trace])), "margin_thr_vs_brake_first5": [round(t["margin_thr_vs_brake"], 4) for t in trace[:5]]}
    if log:
        print(f"  seed {seed} {name}: outcome={out['outcome']} reason={out['failure_reason']} arrive_step={out['arrive_dest_env_step']} env_steps={out['env_steps_after_context']} "
              f"progress={out['progress_m']:.1f} m replans={len(trace)} wall={out['wall_s']:.0f}s replay_ok={replay['ok']}"
              + (f" | k={cfg['ahead']} target={target_progress:.1f} m passed={passed} pass_step={pass_step} artefact={out['n_last_step_brake_artefact']}/{len(trace)} margin_mean={out['margin_thr_vs_brake_mean']:+.3f}" if cfg["goal"] == "fixed" else ""), file=log, flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", default="jepa-latest.pth.tar")
    ap.add_argument("--model-name", default="jepa_wm_driving")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--stimulus", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--configs", default="compact_rolling3,compact_dest,compact_rolling6,compact_rolling10,released_cem_dest")
    ap.add_argument("--episode-seeds", type=int, default=1)
    ap.add_argument("--max-model-steps", type=int, default=400)
    ap.add_argument("--max-wall-per-episode", type=float, default=1200.0)
    ap.add_argument("--full-trace", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bridge-python", default="/opt/conda/envs/metadrive/bin/python")
    # Planner constructor knobs (overridden per config)
    for k, v in (("cem_samples", 32), ("cem_elites", 6), ("cem_iters", 3), ("cem_init_mean", 0.0), ("cem_init_std", 0.5), ("cem_min_std", 0.02), ("tb_min", -1.0), ("tb_max", 0.5), ("steer_max", 0.0)):
        ap.add_argument("--" + k.replace("_", "-"), type=float if isinstance(v, float) else int, default=v)
    args = ap.parse_args(argv)
    args.label = "native"; args.execute_steps = 1
    args.model_tag = args.model_dir.name
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = clr.load_scenes(args.stimulus, args.arm)
    sealed = clr.sealed_seeds(args.stimulus, None)
    seeds = [s for s in args.seeds if s in scenes]
    hit = sorted(set(seeds) & sealed)
    if hit:
        raise SystemExit(f"LABEL-FIRST GUARD: sealed confirmation seeds selected {hit}")
    log = open(args.out / "driver.log", "a")
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] planner diagnostics {args.model_tag} seeds {seeds} configs {args.configs} (hazard-free only)", file=log, flush=True)
    done = set()
    if (args.out / "diag.jsonl").exists():
        for line in (args.out / "diag.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["seed"], r["config"], r["episode_seed"]))
    planner = clr.Planner(args)
    bridge = clr.Bridge(args.bridge_python, Path(__file__).resolve().parent, args.out / "bridge.log")
    init = bridge.call({"cmd": "init", "num_scenarios": max(200_000, max(seeds) + 1), "closed_loop": True})
    (args.out / "bridge_provenance.json").write_text(json.dumps(init, indent=1) + "\n")
    try:
        for seed in seeds:
            ref, frames = reference_rollout(bridge, args, scene := scenes[seed], seed)
            if ref["outcome"] != "success":
                print(f"  seed {seed}: reference NOT reachable ({ref['failure_reason']}); skipped", file=log, flush=True)
                append_jsonl(args.out / "diag.jsonl", clr.json_safe({"config": "reference", "seed": seed, "model": args.model_tag, "outcome": ref["outcome"], "failure_reason": ref["failure_reason"], "episode_seed": 0, "skipped": True}))
                continue
            print(f"  seed {seed} reference always_throttle: arrive_dest env step {ref['arrive_dest_env_step']}, progress {ref['descriptive_progress_m']:.1f} m, {len(frames)} frames, wall {ref['wall_s']:.1f}s", file=log, flush=True)
            if (seed, "reference", 0) not in done:
                append_jsonl(args.out / "diag.jsonl", clr.json_safe({"config": "reference", "model": args.model_tag, "arm": args.arm, "seed": seed, "episode_seed": 0, "outcome": ref["outcome"], "failure_reason": ref["failure_reason"],
                                                                     "arrive_dest_env_step": ref["arrive_dest_env_step"], "env_steps_after_context": ref["env_steps_after_context"], "progress_m": ref["descriptive_progress_m"],
                                                                     "route_completion": ref["native_route_completion"], "wall_s": ref["wall_s"], "replay_ok": ref["replay_ok"], "replay_max_dev": ref["replay_max_dev"], "n_frames": len(frames)}))
            for name in args.configs.split(","):
                for e in range(args.episode_seeds):
                    if (seed, name, e) in done:
                        continue
                    rec = run_config(name, CONFIGS[name], bridge, planner, args, scene, seed, ref, frames, e, log)
                    append_jsonl(args.out / "diag.jsonl", clr.json_safe(rec))
    finally:
        bridge.close()
    rows = [json.loads(l) for l in (args.out / "diag.jsonl").read_text().splitlines() if l.strip()]
    summ: dict[str, Any] = {"model": args.model_tag, "arm": args.arm, "configs": {}}
    for name in ["reference"] + list(CONFIGS):
        sel = [r for r in rows if r["config"] == name and not r.get("skipped")]
        if sel:
            summ["configs"][name] = {"n": len(sel), "n_success": sum(r["outcome"] == "success" for r in sel), "outcomes": {r["seed"]: [r["outcome"], r.get("failure_reason"), r.get("arrive_dest_env_step"), round(r["progress_m"], 1), round(r["wall_s"])] for r in sel},
                                     "spec": CONFIGS.get(name)}
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] DIAG_DONE {args.model_tag}", file=log, flush=True)
    print(json.dumps(summ["configs"], indent=1, default=str))


if __name__ == "__main__":
    main()
