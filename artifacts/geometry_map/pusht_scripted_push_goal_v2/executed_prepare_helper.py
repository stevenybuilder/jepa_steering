#!/usr/bin/env python3
"""Prepare a separate independently seeded, reference-action-reachable Push panel.

CPU physics only. Fifty fixed initial seeds, ten development-only action template
clusters; never screen or replace starts, and never load official held tensors.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def angular_errors(goal, current):
    delta = float(goal)-float(current)
    native = min(abs(delta), 2*math.pi-abs(delta))
    robust = abs(math.atan2(math.sin(delta), math.cos(delta)))
    return {"native_angular_error": native, "robust_wrapped_angular_error": robust,
            "native_negative": native < 0,
            "angle_threshold_disagreement": (native < math.pi/9) != (robust < math.pi/9)}


def validate_manifest(manifest):
    if manifest["panel"] != "pusht_independent_reachable_v1" or manifest["episode_ids"] != list(range(50)):
        raise RuntimeError("Expected the frozen separate50-start panel")
    rows = manifest["starts"]
    if [r["episode"] for r in rows] != list(range(50)):
        raise RuntimeError("Missing or reordered start IDs")
    if len({r["environment_seed"] for r in rows}) != 50 or len({r["planner_seed"] for r in rows}) != 50:
        raise RuntimeError("Duplicate seeds")
    for row in rows:
        if row["environment_seed"] != 2026090700+row["episode"] or row["planner_seed"] != 92600+row["episode"]:
            raise RuntimeError("Unexpected seed rule")
        if row["template_episode"] != row["episode"] % 10:
            raise RuntimeError("Changed predetermined template assignment")
    templates = manifest["development_templates"]
    if [r["episode"] for r in templates] != list(range(10)) or any(r["split"] != "development" for r in templates):
        raise RuntimeError("Only the ten development templates are authorized before tensor access")
    if manifest["action_domain"] != "relative_xy_times_100_unchanged_development_expert_commands":
        raise RuntimeError("Changed native action domain")
    return rows, templates


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


class Contacts:
    def __init__(self, base):
        self.base, self.records = base, []
        self.handler = getattr(base, "collision_handeler", None)
        self.available = self.handler is not None and hasattr(self.handler, "post_solve")

    def __enter__(self):
        if self.available:
            self.original = self.handler.post_solve
            def post_solve(arbiter, space, data):
                def label(body):
                    return "agent" if body is self.base.agent else "block" if body is self.base.block else "static_wall" if body is self.base.space.static_body else "other"
                bodies = [label(shape.body) for shape in arbiter.shapes]
                self.records.append({"bodies": bodies, "agent_block_contact": set(bodies) == {"agent", "block"},
                                     "total_impulse": list(arbiter.total_impulse)})
                if self.original is not None:
                    return self.original(arbiter, space, data)
            self.handler.post_solve = post_solve
        return self

    def __exit__(self, *_):
        if self.available:
            self.handler.post_solve = self.original


def rollout(env, seed, initial, env_info, actions):
    import numpy as np
    import torch
    env.update_env(env_info)
    obs, info = env.prepare(seed, initial, env_info)
    states, observations, proprios, contacts, targets = [np.asarray(info["state"]).copy()], [obs.cpu()], [torch.as_tensor(info["proprio"]).cpu()], [], []
    with Contacts(env.unwrapped) as capture:
        for action in actions:
            capture.records.clear()
            images, _rewards, _dones, infos = env.step_multiple(action.unsqueeze(0))
            if len(images) != 1:
                raise RuntimeError("Native wrapper did not execute exactly one raw action")
            states.append(np.asarray(infos[0]["state"]).copy())
            observations.append(images[0].cpu())
            proprios.append(torch.as_tensor(infos[0]["proprio"]).cpu())
            contacts.append(list(capture.records) if capture.available else None)
            targets.append(np.asarray(env.unwrapped.latest_action).copy())
    return {"states": np.stack(states), "observations": torch.stack(observations).to(torch.uint8),
            "proprios": torch.stack(proprios), "contacts": contacts, "contact_api_available": capture.available,
            "applied_target_xy": np.stack(targets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "project-root", "repo", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    starts, templates = validate_manifest(manifest)  # Before any tensor loading.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from collect_pusht_bank import config
    torch.set_num_threads(2)
    cfg = config(args.repo, args.output_dir)
    cfg.device = "cpu"
    from evals.simu_env_planning.envs.init import make_env
    from evals.simu_env_planning.envs.pusht_gym_wrap import PushTWrapper
    from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
    loaded = {}
    for entry in templates:
        path = args.project_root / entry["path"]
        if sha(path) != entry["sha256"]:
            raise RuntimeError("Development template checksum mismatch")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["split"] != "development" or value["episode"] != entry["episode"]:
            raise RuntimeError("Development template payload identity mismatch")
        actions = value["expert_actions_raw"]
        if tuple(actions.shape) != (30, 2) or not torch.isfinite(actions).all():
            raise RuntimeError("Expected finite original30x2 expert commands")
        loaded[entry["episode"]] = value
    env = make_env(cfg)
    if not env.unwrapped.relative or env.unwrapped.action_scale != 100:
        raise RuntimeError("Native action interpretation differs from frozen generator")
    provenance = {"manifest_sha256": sha(args.manifest), "script_sha256": sha(Path(__file__)),
                  "native_config_sha256": sha(args.repo / manifest["native_config"]),
                  "sample_initial_method": inspect.getsource(PushTWrapper.sample_random_init_goal_states),
                  "native_action_method": inspect.getsource(PushTEnv.step),
                  "native_metric_method": inspect.getsource(PushTWrapper.eval_state),
                  "model_execution": False, "gpu_execution": False,
                  "checkpoint_for_future_baselines": manifest["checkpoint"]}
    if provenance["native_config_sha256"] != manifest["native_config_sha256"]:
        raise RuntimeError("Native configuration hash mismatch")
    write_json(args.output_dir / "PROVENANCE.json", provenance)
    receipts, started = [], time.monotonic()
    try:
        for row in starts:
            t0 = time.monotonic()
            template = loaded[row["template_episode"]]
            raw_initial, _discarded_unreachable_random_goal = env.sample_random_init_goal_states(row["environment_seed"])
            initial = np.asarray(raw_initial).copy()
            initial[4] %= 2*np.pi  # Supported prepare/reset state; confirmed by actual readback below.
            actions, env_info = template["expert_actions_raw"], template["env_info"]
            result = rollout(env, row["environment_seed"], initial, env_info, actions)
            repeat = rollout(env, row["environment_seed"], initial, env_info, actions)
            state_error = float(np.max(np.abs(result["states"]-repeat["states"])))
            pixels_equal = torch.equal(result["observations"], repeat["observations"])
            if state_error > 1e-6 or not pixels_equal:
                raise RuntimeError(f"Episode{row['episode']} exact replay failed: {state_error}, {pixels_equal}")
            goal = result["states"][-1].copy()
            if not np.all((result["states"][:, 4] >= 0) & (result["states"][:, 4] < 2*np.pi)):
                raise RuntimeError("Simulator readback angle not canonical; no silent native-metric patch")
            angular = [angular_errors(goal[4], state[4]) for state in result["states"]]
            native = [env.eval_state(goal, state) for state in result["states"]]
            robust = [bool(np.linalg.norm(goal[:4]-state[:4]) < 20 and metric["robust_wrapped_angular_error"] < np.pi/9)
                      for state, metric in zip(result["states"], angular)]
            disagreements = [i for i, (n, r) in enumerate(zip(native, robust)) if bool(n["success"]) != r]
            if disagreements or any(a["native_negative"] for a in angular):
                raise RuntimeError("Native and robust success differ for canonical readbacks")
            agent_block = [any(c["agent_block_contact"] for c in step) if step is not None else None for step in result["contacts"]]
            value = {"schema_version": 1, "panel": manifest["panel"], "split": "evaluation_preparation_sealed_from_operator_tuning",
                     **row, "source_action_template_cluster": row["template_episode"], "source_template": templates[row["template_episode"]],
                     "reference_actions_are_expert_only_at_original_source": True,
                     "requested_initial_state_unwrapped": raw_initial, "initial_state": initial,
                     "actual_initial_state": result["states"][0], "env_info": env_info,
                     "expert_actions_raw": actions, "expert_observations": result["observations"],
                     "expert_states": result["states"], "expert_proprios": result["proprios"],
                     "legacy_expert_key_semantics": "reference template replay; not an expert trajectory for this new state",
                     "goal_state": goal, "goal_origin": "actual endpoint of exactly30 fixed development-template relative commands",
                     "actual_applied_target_xy": result["applied_target_xy"], "native_contacts": result["contacts"],
                     "contact_api_available": result["contact_api_available"], "agent_block_contact_by_step": agent_block,
                     "native_metric_along_reference": native, "angular_diagnostics": angular,
                     "robust_success_along_reference": robust, "native_robust_disagreement_indices": disagreements,
                     "painted_default_goal_coverage": "not used; not task-goal overlap",
                     "deterministic_replay_max_state_error": state_error, "deterministic_replay_pixels_equal": pixels_equal,
                     "simulator_readback_angles_canonical": True, "source_manifest_sha256": provenance["manifest_sha256"]}
            path = args.output_dir / f"input-{row['episode']:03d}.pt"
            torch.save(value, path)
            receipt = {**row, "path": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
                       "seconds": time.monotonic()-t0, "state_replay_error": state_error,
                       "pixels_exact": pixels_equal, "contact_api_available": result["contact_api_available"],
                       "contact_raw_steps": sum(x is True for x in agent_block),
                       "block_translation_l2": float(np.linalg.norm(goal[2:4]-result["states"][0, 2:4])),
                       "native_robust_disagreements": len(disagreements), "angles_canonical": True}
            receipts.append(receipt)
            print(json.dumps({"event": "reference_input_prepared", "episode": row["episode"], "sha256": receipt["sha256"], "seconds": receipt["seconds"]}), flush=True)
        total = time.monotonic()-started
        write_json(args.output_dir / "DONE.json", {"complete": True, "panel": manifest["panel"], "outputs": receipts,
                   "seconds": total, "cpu_physics_only": True, "model_execution": False, "gpu_execution": False,
                   "provenance_sha256": sha(args.output_dir / "PROVENANCE.json"),
                   "independent_physics_seeds": 50, "source_action_template_clusters": 10,
                   "official_expert_source_panel_unchanged": True,
                   "no_screening_or_replacement": True, "all_starts_retained": True,
                   "distribution_warning": "Expert reference actions are not adapted to independent layouts. Contact may be absent and block goals trivial; this panel is not50expert demonstrations or the released sourcedset benchmark.",
                   "contact_summary_descriptive_only": {"episodes_with_agent_block_contact": sum(r["contact_raw_steps"] > 0 for r in receipts),
                       "episodes_block_translation_over_one_pixel": sum(r["block_translation_l2"] > 1 for r in receipts)},
                   "gpu_baseline_launch_authorized": False})
    except Exception as exc:
        write_json(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc), "outputs": receipts})
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
