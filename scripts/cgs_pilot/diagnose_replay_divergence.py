#!/usr/bin/env python3
"""Why does a fresh process not reproduce a saved cell's final_state exactly?

Runs, for one cell, in ONE fresh process:
  A. fresh env -> restore_replay_state -> replay            (validator path)
  B. same env, repeat A                                      (within-process determinism)
  C. fresh env -> np.random.seed(seed); env.reset(); approach_egg(env)
     -> restore -> replay                                    (generator's history)
and reports max |final_state - saved| for each, plus per-step first divergence
index and whether contact/force labels replicate. Run it twice in two processes
to separate history-dependent hidden state (A==B, C exact, A!=C, A stable across
processes) from nondeterministic numerics (A differs across processes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robocasa_contact_pilot as gen  # noqa: E402


def replay_states(env, model_xml, start, actions, steps_per):
    gen.restore_replay_state(env, model_xml, start)
    states = [env.sim.get_state().flatten().copy()]
    max_force, contacts = 0.0, 0
    for a in actions:
        sim_a = gen.model_to_sim_action(a, env.action_dim)
        for _ in range(steps_per):
            env.step(sim_a)
            m = gen.contact_metrics(env)
            max_force = max(max_force, m["max_normal_force_n"])
            contacts += m["egg_robot_contact_count"]
            states.append(env.sim.get_state().flatten().copy())
    return np.stack(states), max_force, contacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus-dir", type=Path, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-dummy", type=int, default=6)
    ap.add_argument("--actions-from-candidate-sequence", action="store_true", help="rebuild float64 actions via gen.candidate_sequence instead of the saved float32 array")
    ap.add_argument("--mode", choices=["replay", "regenerate", "with_control", "prior_cells", "dummy_restores"], default="replay")
    args = ap.parse_args()
    rows = {json.loads(l)["cell_id"]: json.loads(l) for l in (args.stimulus_dir / "manifest.jsonl").read_text().splitlines()}
    row = rows[args.cell]
    seed = int(row["seed"])
    cell = np.load(args.stimulus_dir / row["artifact"])
    start = np.asarray(cell["initial_state"], dtype=np.float64)
    saved_final = np.asarray(cell["final_state"], dtype=np.float64)
    actions = np.asarray(cell["model_actions"], dtype=np.float32)
    if args.actions_from_candidate_sequence:
        rebuilt = gen.candidate_sequence(int(row["candidate_action"]), int(row["horizon"]))
        assert np.array_equal(rebuilt.astype(np.float32), actions), "candidate_sequence does not match saved actions"
        actions = rebuilt
    steps_per = int(row.get("sim_steps_per_model_step", 5))
    out = {"cell": args.cell, "seed": seed, "saved_contacts": row["egg_robot_contact_count"], "saved_force": row["max_normal_force_n"], "mode": args.mode}

    if args.mode == "regenerate":
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="regen_"))
        env = gen.make_env(seed)
        try:
            rows_new, summary = gen.generate_seed(env, seed, tmp, int(row["horizon"]), steps_per)
        finally:
            env.close()
        new = np.load(tmp / row["artifact"])
        out["regen_final_err_vs_saved"] = float(np.max(np.abs(np.asarray(new["final_state"]) - saved_final)))
        out["regen_initial_err_vs_saved"] = float(np.max(np.abs(np.asarray(new["initial_state"]) - start)))
        fr_new = np.asarray(new["true_future_frames"]).astype(np.int16); fr_old = np.asarray(cell["true_future_frames"]).astype(np.int16)
        out["regen_frame_max_abs"] = int(np.abs(fr_new - fr_old).max())
        out["regen_frame_diff_px_per_step"] = [int((np.abs(fr_new[i] - fr_old[i]).max(axis=-1) > 0).sum()) for i in range(len(fr_new))]
        newrow = [r for r in rows_new if r["cell_id"] == args.cell][0]
        out["regen_force"] = newrow["max_normal_force_n"]; out["regen_contacts"] = newrow["egg_robot_contact_count"]
        out["regen_force_did"] = summary["force_did_n"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=1))
        return

    if args.mode == "with_control":
        env = gen.make_env(seed)
        np.random.seed(seed)
        env.reset()
        gen.approach_egg(env)
        base_state = env.sim.get_state().flatten().copy()
        egg_joint = env.objects[gen.TARGET_OBJECT].joints[0]
        egg_qpos = np.asarray(env.sim.data.get_joint_qpos(egg_joint), dtype=np.float64).copy()
        gripper_position = gen.proprio(env)[:3].astype(np.float64)
        xml_w = env.sim.model.get_xml()
        vis = gen.target_visibility(env)
        gen.choose_control_position(env, base_state, egg_qpos, gripper_position, np.asarray(vis["target_centroid_xy"], dtype=np.float64))
        env.sim.set_state_from_flattened(base_state.copy())
        gen.set_free_joint_pose(env, egg_qpos[:3].copy(), egg_qpos[3:7])
        hz = env.sim.get_state().flatten().copy()
        out["rebuilt_hazard_state_err_vs_saved_initial"] = float(np.max(np.abs(hz - start)))
        gen.render_rgb(env); gen.proprio(env)
        sW, fW, cW = replay_states(env, xml_w, start, actions, steps_per)
        env.close()
        out["W_with_control_search"] = {"final_err_vs_saved": float(np.max(np.abs(sW[-1] - saved_final))), "force": fW, "contacts": cW}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=1))
        return

    if args.mode in ("prior_cells", "dummy_restores"):
        env = gen.make_env(seed)
        np.random.seed(seed)
        env.reset()
        xml_p = env.sim.model.get_xml()
        if args.mode == "prior_cells":
            order = [(0, 0), (0, 1), (1, 0)]
            for h, a in order:
                r = rows[f"{row['pair_id']}__h{h}a{a}"]
                c = np.load(args.stimulus_dir / r["artifact"])
                for _ in range(2):
                    gen.run_candidate(env, xml_p, np.asarray(c["initial_state"], dtype=np.float64), np.asarray(c["model_actions"], dtype=np.float32), steps_per)
            gen.run_candidate(env, xml_p, start, actions, steps_per)  # first replay of this cell, as in generator
        else:
            for _ in range(args.n_dummy):
                gen.restore_replay_state(env, xml_p, start)
        sP, fP, cP = replay_states(env, xml_p, start, actions, steps_per)
        env.close()
        out[args.mode] = {"final_err_vs_saved": float(np.max(np.abs(sP[-1] - saved_final))), "force": fP, "contacts": cP, "n_dummy": args.n_dummy}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=1))
        return

    env = gen.make_env(seed)
    env.reset()
    xml = env.sim.model.get_xml()
    sA, fA, cA = replay_states(env, xml, start, actions, steps_per)
    sB, fB, cB = replay_states(env, xml, start, actions, steps_per)
    env.close()

    env = gen.make_env(seed)
    np.random.seed(seed)
    env.reset()
    gen.approach_egg(env)
    xml_c = env.sim.model.get_xml()
    sC, fC, cC = replay_states(env, xml_c, start, actions, steps_per)
    sD, fD, cD = replay_states(env, xml_c, start, actions, steps_per)
    env.close()

    def rep(name, s, f, c):
        err = float(np.max(np.abs(s[-1] - saved_final)))
        out[name] = {"final_err_vs_saved": err, "force": f, "contacts": c,
                     "label_replicates": bool((c > 0) == (row["egg_robot_contact_count"] > 0))}

    rep("A_fresh", sA, fA, cA)
    rep("B_fresh_repeat", sB, fB, cB)
    rep("C_after_generator_history", sC, fC, cC)
    rep("D_after_history_repeat", sD, fD, cD)
    out["A_vs_B_max"] = float(np.max(np.abs(sA - sB)))
    out["A_vs_C_max"] = float(np.max(np.abs(sA - sC)))
    out["C_vs_D_max"] = float(np.max(np.abs(sC - sD)))
    d = np.max(np.abs(sA - sC), axis=1)
    out["A_vs_C_first_divergent_step"] = int(np.argmax(d > 0)) if np.any(d > 0) else -1
    out["A_vs_C_per_step"] = [float(x) for x in d]
    out["xml_same_A_C"] = xml == xml_c
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "A_vs_C_per_step"}, indent=1))


if __name__ == "__main__":
    main()
