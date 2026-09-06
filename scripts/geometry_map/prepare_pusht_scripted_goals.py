#!/usr/bin/env python3
"""One frozen contact-seeking scripted-goal correction; CPU-only, all50 retained."""
from __future__ import annotations
import argparse
import inspect
import json
import math
from pathlib import Path
import time
from prepare_pusht_independent import Contacts, angular_errors, rollout, sha, write_json


def unit_direction(agent, block):
    delta = [float(block[i])-float(agent[i]) for i in range(2)]
    length = math.hypot(*delta)
    return [x/length for x in delta] if length > 1e-12 else [1., 0.]


def command(agent, block, direction):
    aim = [min(492., max(20., float(block[i])+30.*direction[i])) for i in range(2)]
    raw = [min(.25, max(-.25, (aim[i]-float(agent[i]))/100.)) for i in range(2)]
    return raw, aim


def validate(manifest):
    if manifest["panel"] != "pusht_scripted_push_goal_v2" or manifest["episode_ids"] != list(range(50)):
        raise RuntimeError("Wrong panel or incomplete50fixedstarts")
    rows = manifest["starts"]
    if [r["episode"] for r in rows] != list(range(50)):
        raise RuntimeError("Changed episode ordering")
    if any(r["environment_seed"] != 2026090700+r["episode"] or r["planner_seed"] != 92600+r["episode"] for r in rows):
        raise RuntimeError("Independent initial seeds must equal preservedv1")
    expected = {"steps": 30, "through_block_offset": 30, "aim_bounds": [20, 492], "relative_scale": 100, "raw_bounds": [-.25, .25]}
    if manifest["controller"] != expected:
        raise RuntimeError("Frozen controller changed")
    return rows


def generated_rollout(env, seed, initial):
    import numpy as np
    import torch
    env_info = {"shape": "T"}
    env.update_env(env_info)
    obs, info = env.prepare(seed, initial, env_info)
    states, frames, proprios = [np.asarray(info["state"]).copy()], [obs.cpu()], [torch.as_tensor(info["proprio"]).cpu()]
    base = env.unwrapped
    if not np.allclose(states[0][:2], base.agent.position, atol=1e-5) or not np.allclose(states[0][2:4], base.block.position, atol=1e-5):
        raise RuntimeError("Native state body-coordinate ordering differs")
    direction = unit_direction(base.agent.position, base.block.position)
    actions, targets, aims, contacts, intents, ideal_targets = [], [], [], [], [], []
    with Contacts(base) as capture:
        for _step in range(30):
            raw, aim = command(base.agent.position, base.block.position, direction)
            action = torch.tensor(raw, dtype=torch.float32)
            # Match the actual native Vec2d + numpyfloat32 arithmetic. Converting
            # Vec2d to a float64array first changes NumPy2 promotion/rounding.
            expected_target = np.asarray(base.agent.position + np.array(action)*base.action_scale)
            ideal_target = np.asarray(base.agent.position, dtype=np.float64)+100*np.asarray(raw, dtype=np.float64)
            capture.records.clear()
            images, _rewards, _dones, infos = env.step_multiple(action.unsqueeze(0))
            if len(images) != 1 or not np.array_equal(np.asarray(base.latest_action), expected_target):
                raise RuntimeError(f"Actual native relative control differs: raw_step={_step+1}, images={len(images)}, command={action.tolist()}, expected_target={expected_target.tolist()}, actual_target={list(base.latest_action)}")
            states.append(np.asarray(infos[0]["state"]).copy())
            frames.append(images[0].cpu())
            proprios.append(torch.as_tensor(infos[0]["proprio"]).cpu())
            actions.append(action)
            intents.append(raw)
            ideal_targets.append(ideal_target)
            aims.append(aim)
            targets.append(np.asarray(base.latest_action).copy())
            contacts.append(list(capture.records) if capture.available else None)
    return {"states": np.stack(states), "observations": torch.stack(frames).to(torch.uint8), "proprios": torch.stack(proprios),
            "actions": torch.stack(actions), "contacts": contacts, "contact_api_available": capture.available,
            "initial_push_direction": direction, "clipped_aim_xy": np.asarray(aims), "applied_target_xy": np.stack(targets),
            "ideal_raw_command_float64": np.asarray(intents), "ideal_target_float64": np.stack(ideal_targets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "repo", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    starts = validate(manifest)
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
    env = make_env(cfg)
    base = env.unwrapped
    if base.window_size != 512 or not base.relative or base.action_scale != 100:
        raise RuntimeError("Native512arena/relativeXYx100 interpretation changed")
    provenance = {"script_sha256": sha(Path(__file__)), "helper_sha256": sha(Path(inspect.getfile(rollout))),
                  "manifest_sha256": sha(args.manifest), "native_config_sha256": sha(args.repo/manifest["native_config"]),
                  "native_init_source": inspect.getsource(PushTEnv.__init__), "native_action_source": inspect.getsource(PushTEnv.step),
                  "native_observation_source": inspect.getsource(PushTEnv._get_obs),
                  "native_metric_source": inspect.getsource(PushTWrapper.eval_state),
                  "native_random_state_source": inspect.getsource(PushTWrapper.sample_random_init_goal_states),
                  "checkpoint_for_future_baselines": manifest["checkpoint"], "arena_size": 512,
                  "declared_action_space": str(base.action_space),
                  "declared_box_caveat": "Native Box0..512 is absolute-looking but relative-mode step does not clip; actual native controls are relativeXYx100. Controller rawbounds±0.25 are explicit generator limits, not an invented environment clamp.",
                  "model_execution": False, "gpu_execution": False}
    if provenance["native_config_sha256"] != manifest["native_config_sha256"]:
        raise RuntimeError("Nativeconfig mismatch")
    write_json(args.output_dir/"PROVENANCE.json", provenance)
    receipts, start = [], time.monotonic()
    try:
        for row in starts:
            t0 = time.monotonic()
            requested, _discarded = env.sample_random_init_goal_states(row["environment_seed"])
            initial = np.asarray(requested).copy()
            initial[4] %= 2*np.pi
            result = generated_rollout(env, row["environment_seed"], initial)
            repeat = rollout(env, row["environment_seed"], initial, {"shape":"T"}, result["actions"])
            state_error = float(np.max(np.abs(result["states"]-repeat["states"])))
            pixels_equal = torch.equal(result["observations"], repeat["observations"])
            if state_error > 1e-6 or not pixels_equal:
                raise RuntimeError(f"Exact cached-command replay failed episode{row['episode']}: {state_error}, {pixels_equal}")
            states, goal = result["states"], result["states"][-1].copy()
            if not np.all((states[:,4] >= 0) & (states[:,4] < 2*np.pi)):
                raise RuntimeError("Uncanonical simulator readback; no silentnative metric patch")
            angular = [angular_errors(goal[4], s[4]) for s in states]
            native = [env.eval_state(goal, s) for s in states]
            robust = [bool(np.linalg.norm(goal[:4]-s[:4])<20 and a["robust_wrapped_angular_error"]<np.pi/9) for s,a in zip(states,angular)]
            disagreements = [i for i,(a,b) in enumerate(zip(native,robust)) if bool(a["success"]) != b]
            if disagreements or any(a["native_negative"] for a in angular):
                raise RuntimeError("Native metric validity disagreement")
            contacts = [any(c["agent_block_contact"] for c in step) if step is not None else None for step in result["contacts"]]
            value = {"schema_version":2,"panel":manifest["panel"],"split":"evaluation",**row,
                     "source_trajectory":f"scripted-physics-seed-{row['environment_seed']}",
                     "source_kind":"independent nativeinitialization with fixedphysical-feedback scripted reference; no expertvideo",
                     "source_action_template_clusters":0,"controller_family_count":1,
                     "requested_initial_state_unwrapped":requested,"initial_state":initial,"actual_initial_state":states[0],"env_info":{"shape":"T"},
                     "expert_actions_raw":result["actions"],"expert_observations":result["observations"],"expert_states":states,"expert_proprios":result["proprios"],
                     "legacy_expert_key_semantics":"scripted reference rollout; names preserved only for native baselineinput compatibility",
                     "goal_state":goal,"goal_origin":"actual30stependpoint of frozen physical-feedback controller; all30commands cached",
                     "initial_push_direction":result["initial_push_direction"],"clipped_aim_xy":result["clipped_aim_xy"],
                     "actual_applied_target_xy":result["applied_target_xy"],"native_contacts":result["contacts"],"agent_block_contact_by_step":contacts,
                     "ideal_raw_command_float64":result["ideal_raw_command_float64"],"ideal_target_float64":result["ideal_target_float64"],
                     "native_applied_target_verification":"exact Vec2d+numpyfloat32 native conversion, not float64ideal target; cached command replay still exact",
                     "contact_api_available":result["contact_api_available"],"native_metric_along_reference":native,"angular_diagnostics":angular,
                     "robust_success_along_reference":robust,"native_robust_disagreement_indices":disagreements,
                     "deterministic_replay_max_state_error":state_error,"deterministic_replay_pixels_equal":pixels_equal,
                     "simulator_readback_angles_canonical":True,"source_manifest_sha256":provenance["manifest_sha256"],
                     "painted_default_goal_coverage":"unused; not taskgoaloverlap"}
            path = args.output_dir/f"input-{row['episode']:03d}.pt"
            torch.save(value,path)
            receipt = {**row,"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size,"seconds":time.monotonic()-t0,
                       "state_replay_error":state_error,"pixels_exact":pixels_equal,"contact_raw_steps":sum(x is True for x in contacts),
                       "contact_api_available":result["contact_api_available"],"block_translation_l2":float(np.linalg.norm(goal[2:4]-states[0,2:4])),
                       "block_rotation_wrapped":angular_errors(goal[4],states[0,4])["robust_wrapped_angular_error"],
                       "joint_agent_block_position_displacement":float(np.linalg.norm(goal[:4]-states[0,:4])),
                       "native_robust_disagreements":len(disagreements),"angles_canonical":True,
                       "raw_command_max_abs":float(result["actions"].abs().max())}
            receipts.append(receipt)
            print(json.dumps({"event":"scripted_input_prepared","episode":row["episode"],"sha256":receipt["sha256"],"seconds":receipt["seconds"]}),flush=True)
        write_json(args.output_dir/"DONE.json",{"complete":True,"panel":manifest["panel"],"outputs":receipts,"seconds":time.monotonic()-start,
                   "cpu_physics_only":True,"model_execution":False,"gpu_execution":False,"gpu_baseline_launch_authorized":False,
                   "provenance_sha256":sha(args.output_dir/"PROVENANCE.json"),"independent_physics_seeds":50,"controller_family_count":1,
                   "official_expert_panel_unchanged":True,"v1_panel_preserved":True,"no_screening_or_replacement":True,"all_starts_retained":True,
                   "generator_search":"exactly one main-authorizedv2 correction; no further automatic search",
                   "contact_summary_descriptive_only":{"episodes_with_agent_block_contact":sum(r["contact_raw_steps"]>0 for r in receipts),
                       "episodes_block_translation_over_one_pixel":sum(r["block_translation_l2"]>1 for r in receipts)},
                   "comparability":"separate scripted-push-goal panel, not released expert-sourced benchmark or50expertvideos"})
    except Exception as exc:
        write_json(args.output_dir/"FAILED.json",{"complete":False,"error":str(exc),"outputs":receipts})
        raise
    finally:
        env.close()


if __name__ == "__main__":main()
