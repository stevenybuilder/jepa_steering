#!/usr/bin/env python3
"""Decide whether a `deterministic_replay=False` cell is renderer jitter or physics.

For each requested cell this script replays the frozen episode twice with the
generator's own `restore_replay_state`/`run_candidate` machinery and records:

* the flattened MuJoCo state after EVERY simulator step (not only the final
  state), so any physics divergence at any time is caught;
* per-frame pixel differences between the two replays (count, max magnitude,
  channel/location), so the size of the image discrepancy is documented;
* N repeated renders of the SAME state without stepping, so pure renderer
  nondeterminism is measured independently of physics.

The verdict per cell is `render_only` when all per-step states are bit-identical
and repeated same-state renders show the same kind of jitter; anything else is
`physics_or_unknown`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robocasa_contact_pilot as gen  # noqa: E402


def pixel_report(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mask = d.max(axis=-1) > 0
    coords = np.argwhere(mask)
    return {
        "differing_pixels": int(mask.sum()),
        "max_abs_diff": int(d.max()),
        "total_pixels": int(mask.size),
        "example_locations_yx": coords[:8].tolist(),
    }


def replay_with_step_states(env, model_xml, start_state, actions, steps_per):
    gen.restore_replay_state(env, model_xml, start_state)
    states = [env.sim.get_state().flatten().copy()]
    frames = [gen.render_rgb(env)]
    for model_action in actions:
        sim_action = gen.model_to_sim_action(model_action, env.action_dim)
        for _ in range(steps_per):
            env.step(sim_action)
            states.append(env.sim.get_state().flatten().copy())
        frames.append(gen.render_rgb(env))
    return np.stack(states), np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="*", default=None, help="cell_ids; default = all non-deterministic cells")
    ap.add_argument("--repeat-renders", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.stimulus_dir / "manifest.jsonl").read_text().splitlines()]
    if args.cells:
        rows = [r for r in rows if r["cell_id"] in set(args.cells)]
    else:
        rows = [r for r in rows if not r["deterministic_replay"]]
    report = []
    for row in rows:
        seed = int(row["seed"])
        env = gen.make_env(seed)
        try:
            cell = np.load(args.stimulus_dir / row["artifact"])
            start_state = np.asarray(cell["initial_state"], dtype=np.float64)
            actions = np.asarray(cell["model_actions"], dtype=np.float32)
            model_xml = env.sim.model.get_xml()
            steps_per = int(row.get("sim_steps_per_model_step", 5))

            s1, f1 = replay_with_step_states(env, model_xml, start_state, actions, steps_per)
            s2, f2 = replay_with_step_states(env, model_xml, start_state, actions, steps_per)
            step_state_err = float(np.max(np.abs(s1 - s2)))
            per_frame = [pixel_report(f1[i], f2[i]) for i in range(len(f1))]

            # Same-state repeated renders: no stepping between renders.
            gen.restore_replay_state(env, model_xml, start_state)
            renders = [gen.render_rgb(env) for _ in range(args.repeat_renders)]
            same_state = [pixel_report(renders[0], renders[i]) for i in range(1, len(renders))]
            # Also compare the saved first-run frames to this replay.
            saved = np.concatenate([np.asarray(cell["context_frames"])[-1:], np.asarray(cell["true_future_frames"])])
            vs_saved = [pixel_report(saved[i], f1[i]) for i in range(len(saved))]

            render_only = (
                step_state_err == 0.0
                and all(p["max_abs_diff"] <= 1 for p in per_frame)
                and all(p["max_abs_diff"] <= 1 for p in same_state)
                and all(p["max_abs_diff"] <= 1 for p in vs_saved)
            )
            report.append(
                {
                    "cell_id": row["cell_id"],
                    "seed": seed,
                    "per_step_state_max_abs_err": step_state_err,
                    "n_steps_compared": int(len(s1)),
                    "replay_vs_replay_frames": per_frame,
                    "same_state_repeated_renders": same_state,
                    "saved_vs_replay_frames": vs_saved,
                    "verdict": "render_only" if render_only else "physics_or_unknown",
                }
            )
            print(json.dumps(report[-1]))
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
