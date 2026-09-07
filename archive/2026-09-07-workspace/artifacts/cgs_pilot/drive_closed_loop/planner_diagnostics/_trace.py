import json, sys, numpy as np
for m in ("armA_seed0", "armB_seed0"):
    p = f"/root/cgs-pilot/artifacts/drive_closed_loop/planner_diagnostics/{m}/diag.jsonl"
    try: rows = [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: continue
    for r in rows:
        if r["config"] == "reference": print(f"{m} seed {r['seed']} REFERENCE arrive {r['arrive_dest_env_step']} progress {r['progress_m']:.1f}"); continue
        print(f"\n{m} seed {r['seed']} {r['config']}: outcome={r['outcome']} reason={r['failure_reason']} arrive={r['arrive_dest_env_step']} steps={r['env_steps_after_context']} progress={r['progress_m']:.1f} replans={r['n_replans']} wall={r['wall_s']:.0f}s")
        tr = r["trace"]
        show = list(range(min(12, len(tr)))) + [i for i in (20, 40, 80, 160, 320) if i < len(tr)]
        for i in show:
            t = tr[i]
            prof = t.get("ref_profile_encoder_only"); prof = [round(x) for x in prof] if prof else None
            print(f"  k{i:3d} g{t['goal_index']:3d} v={t['speed_after']:5.2f} prog={t['progress_after_m']:6.1f} exec={t['executed_tb'][0]:+.2f} cost brake={t['cost_brake']:7.0f} thr={t['cost_throttle']:7.0f} final={t['cost_final']:7.0f} static={t.get('cost_static_encoder_only', float('nan')):7.0f} refprof={prof}")
        # summary stats over the episode
        cb = np.array([t["cost_brake"] for t in tr]); ct = np.array([t["cost_throttle"] for t in tr]); ex = np.array([t["executed_tb"][0] for t in tr]); v = np.array([t["speed_after"] for t in tr])
        print(f"  frac replans brake<throttle: {(cb < ct).mean():.2f}; mean exec tb {ex.mean():+.3f}; frac exec<0 {(ex<0).mean():.2f}; frac exec<=-0.5 {(ex<=-0.5).mean():.2f}; mean speed {v.mean():.2f}; replans at v<0.5: {(v<0.5).mean():.2f}")
        if "cost_static_encoder_only" in tr[0]:
            cs = np.array([t["cost_static_encoder_only"] for t in tr]); print(f"  static vs brake: mean(brake-static)={np.mean(cb-cs):+.1f}; static vs throttle: mean(thr-static)={np.mean(ct-cs):+.1f}; frac static<min(brake,thr): {(cs < np.minimum(cb, ct)).mean():.2f}")
