#!/usr/bin/env python3
"""Freeze pointers to original baselines; package every development replan.

The inventory phase uses only stdlib, receipt metadata, and byte hashes. Held
banks are never deserialized. Packaging runs on CPU and loads development only.
This creates replay inputs, not steered results or a new state-selection rule.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def relative(root, path):
    return str(path.resolve().relative_to(root.resolve()))


def receipt(path):
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise RuntimeError(f"Incomplete receipt: {path}")
    return value


def verify_entry(root, directory, entry):
    path = directory / entry["path"]
    if path.parent.resolve() != directory.resolve():
        raise RuntimeError("Receipt path escapes source directory")
    actual = sha(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"Hash mismatch before tensor access: {path}")
    return {"path": relative(root, path), "sha256": actual, "bytes": path.stat().st_size}


def expected_split(task, episode):
    maximum, boundary = (24, 12) if task == "reach_wall" else (21, 10)
    if not 0 <= episode < maximum:
        raise RuntimeError("Unexpected baseline episode")
    return "development" if episode < boundary else "evaluation"


def pointer(bank, *keys, index=None, selection=None):
    result = {"bank_path": bank["path"], "bank_sha256": bank["sha256"], "keys": list(keys)}
    if index is not None:
        result["index"] = index
    if selection is not None:
        result["selection"] = selection
    return result


def inventory(args):
    root = args.project_root.resolve()
    reach_manifest = json.loads(args.reach_manifest.read_text())
    if reach_manifest["episode_count"] != 24 or reach_manifest["seed_rules"] != {
        "environment_seed": "2026090500 + episode_id", "planner_seed": "90500 + episode_id"}:
        raise RuntimeError("Reach frozen manifest changed")
    prepared = json.loads(args.prepared_manifest.read_text())
    prepared_done = receipt(args.prepared_manifest.parent / "DONE.json")
    if prepared["episodes"] != prepared_done["episodes"]:
        raise RuntimeError("Prepared input manifest and receipt disagree")
    inputs = {}
    for entry in prepared["episodes"]:
        episode = int(entry["episode"])
        if episode in inputs or entry["split"] != expected_split("pusht", episode):
            raise RuntimeError("Prepared input duplicate or split mismatch")
        inputs[episode] = {**verify_entry(root, args.prepared_manifest.parent, entry),
                           "episode": episode, "split": entry["split"], "source_trajectory": int(entry["source_trajectory"])}
    if set(inputs) != set(range(21)) or len({r["source_trajectory"] for r in inputs.values()}) != 21:
        raise RuntimeError("Expected 21 disjoint official source trajectories")
    checkpoints = json.loads(args.checkpoints.read_text())
    banks, source_receipts = [], []
    for task, directories, expected_model in (("reach_wall", args.reach_dirs, "jepa_wm_metaworld"),
                                               ("pusht", args.pusht_dirs, "jepa_wm_pusht")):
        seen = set()
        for directory in directories:
            done_path = directory / "DONE.json"
            done = receipt(done_path)
            source_receipts.append({"path": relative(root, done_path), "sha256": sha(done_path)})
            provenance = done["model_provenance"]
            model = checkpoints[task]
            if provenance["model_name"] != expected_model or Path(provenance["checkpoint"]).name != model["filename"]:
                raise RuntimeError("Checkpoint model mismatch")
            if model["snapshot_revision"] not in provenance["checkpoint"]:
                raise RuntimeError("Checkpoint snapshot revision mismatch")
            for entry in done["outputs"]:
                # Whitelist fields: never copy or use success, reward, distance,
                # duration, or held replan-count outcome metadata from receipts.
                episode = int(entry["episode"])
                split = expected_split(task, episode)
                if episode in seen or ("split" in entry and entry["split"] != split):
                    raise RuntimeError("Duplicate baseline episode or split mismatch")
                seen.add(episode)
                row = {**verify_entry(root, directory, entry), "task": task, "episode": episode,
                       "split": split, "sealed": split == "evaluation", "arm": "unsteered_frozen_jepa_wm",
                       "source_receipt": relative(root, done_path), "source_receipt_sha256": sha(done_path),
                       "model": {"name": expected_model, "checkpoint_filename": model["filename"],
                                 "checkpoint_sha256": model["sha256"], "snapshot_revision": model["snapshot_revision"],
                                 "hash_verification": model["verification"], "config": Path(provenance["config"]).name},
                       "tensor_opened_for_packaging": False}
                if task == "reach_wall":
                    seeds = (2026090500 + episode, 90500 + episode)
                    if (int(entry["environment_seed"]), int(entry["planner_seed"])) != seeds:
                        raise RuntimeError("Reach seeds disagree with original frozen rules")
                    row.update({"environment_seed": seeds[0], "planner_seed": seeds[1],
                                "seed_verification": "original receipt and frozen manifest",
                                "replay_input_sha256": row["sha256"],
                                "input_provenance": "original reset/expert goal and replay inputs embedded in bank; no separate raw input artifact recorded",
                                "config_sha256": done["config_sha256"]})
                else:
                    row.update({"prepared_input": inputs[episode],
                                "environment_seed": 2026090600 + episode, "planner_seed": 91600 + episode,
                                "seed_verification": "original collector seed rules; held payload verification deferred" if split == "evaluation" else "original collector seed rules; development payload verification pending",
                                "replay_input_sha256": inputs[episode]["sha256"],
                                "config_sha256": prepared["config_sha256"]})
                banks.append(row)
        if seen != set(range(24 if task == "reach_wall" else 21)):
            raise RuntimeError(f"Incomplete {task} baseline coverage: {sorted(seen)}")
    banks.sort(key=lambda r: (r["task"], r["episode"]))
    value = {"schema_version": 1, "complete": True, "stage": "hash-verified source inventory; no tensors opened",
             "created_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root),
             "banks": banks, "source_receipts": source_receipts, "prepared_inputs": list(inputs.values()),
             "manifests": [{"path": relative(root, p), "sha256": sha(p)} for p in (args.reach_manifest, args.prepared_manifest)],
             "checkpoint_receipt": {"path": relative(root, args.checkpoints), "sha256": sha(args.checkpoints)},
             "source_counts": {"reach_baseline": 24, "pusht_baseline": 21, "prepared_official_inputs_not_baselines": 21},
             "split_counts": {"development": 22, "sealed_evaluation": 23},
             "scope": "all original baseline episodes; no outcome-dependent selection, no interventions"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_new(args.output_dir / "source_inventory.json", value)
    transfer = []
    for row in banks:
        if row["sealed"]:
            continue
        transfer.append(row["path"])
        if row["task"] == "pusht":
            transfer.append(row["prepared_input"]["path"])
    with (args.output_dir / "development-transfer-files.txt").open("x") as stream:
        stream.write("\n".join(sorted(set(transfer))) + "\n")
    print(json.dumps({"stage": "inventory_complete", "baselines": len(banks), "development": 22, "sealed": 23}), flush=True)


def load_development(root, row, loader=None):
    # The inventory split is an access gate, before hash or deserialization.
    if row["sealed"] or row["split"] != "development" or expected_split(row["task"], int(row["episode"])) != "development":
        raise RuntimeError("Refusing to open sealed evaluation tensor")
    path = root / row["path"]
    if sha(path) != row["sha256"]:
        raise RuntimeError("Development bank checksum mismatch")
    if loader is None:
        import torch
        loader = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    bank = loader(path)
    if int(bank["episode"]) != int(row["episode"]) or bank["arm"] != "unsteered_frozen_jepa_wm":
        raise RuntimeError("Bank identity mismatch")
    if (int(bank["environment_seed"]), int(bank["planner_seed"])) != (row["environment_seed"], row["planner_seed"]):
        raise RuntimeError("Original seed mismatch")
    return bank


def replay_prefixes(replans, action_count, task):
    offset = 1 if task == "reach_wall" else 0
    prefix, result = 0, []
    for index, row in enumerate(replans):
        count = int(row["executed_action_count"])
        elapsed = int(row["elapsed_steps"])
        if count <= 0 or elapsed != offset + prefix:
            raise RuntimeError(f"Replay elapsed/prefix mismatch at replan {index}")
        if prefix + count > action_count:
            raise RuntimeError("Replay prefix exceeds cached actions")
        result.append((prefix, count, elapsed))
        prefix += count
    if prefix != action_count:
        raise RuntimeError("Executed chunks do not cover cached action history")
    return result


def numeric(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise RuntimeError("Nonfinite development data")
    return value


def motion_summary(actions, xyz_dims):
    import numpy as np
    translation = numeric(actions)[:, :xyz_dims]
    net = translation.sum(0)
    norm = float(np.linalg.norm(net))
    return {"raw_steps": len(translation), "axes": "XYZ" if xyz_dims == 3 else "XY",
            "raw_command_units_not_physical_displacement": True,
            "applied_environment_controls_saved": False,
            "interpretation": "planner commands submitted to environment; environment clipping/scaling can change applied controls",
            "maximum_absolute_raw_command": float(np.abs(numeric(actions)).max()),
            "translation_net": net.tolist(), "translation_net_norm": norm,
            "translation_unit_direction": (net / norm).tolist() if norm > 1e-12 else None,
            "translation_frobenius_norm": float(np.linalg.norm(translation)),
            "all_action_frobenius_norm": float(np.linalg.norm(numeric(actions)))}


def reach_conditions(bank, index):
    import numpy as np
    from protocol import box_signed_distance, segment_intersects_box
    replan = bank["replans"][index]
    hand = numeric(replan["observation_proprio"]).reshape(-1)[:3]
    goal = numeric(replan["simulator"]["task_state"]["_target_pos"]).reshape(3)
    initial = numeric(bank["replans"][0]["observation_proprio"]).reshape(-1)[:3]
    distance = float(np.linalg.norm(hand-goal))
    result = {"source": "actual cached replan observation_proprio; generic initial state is stale before warmup",
              "hand_xyz": hand.tolist(), "task_goal_xyz": goal.tolist(), "goal_distance": distance,
              "cumulative_goal_progress": float(np.linalg.norm(initial-goal))-distance,
              "units": "meters", "wall_point_signed_distance": float(box_signed_distance(hand[None])[0]),
              "hand_to_goal_proxy_margin0": bool(segment_intersects_box(hand[None], goal[None], 0)[0]),
              "hand_to_goal_proxy_margin0_03": bool(segment_intersects_box(hand[None], goal[None], .03)[0]),
              "raw_step_realized_path_available": False, "body_collision_checked": False}
    if index+1 < len(bank["replans"]):
        end = numeric(bank["replans"][index+1]["observation_proprio"]).reshape(-1)[:3]
        result.update({"next_cached_replan_hand_xyz": end.tolist(), "observed_chunk_displacement_xyz": (end-hand).tolist(),
                       "observed_chunk_goal_progress": distance-float(np.linalg.norm(end-goal)),
                       "sparse_waypoint_chord_margin0": bool(segment_intersects_box(hand[None], end[None], 0)[0]),
                       "sparse_waypoint_chord_margin0_03": bool(segment_intersects_box(hand[None], end[None], .03)[0]),
                       "chord_interpretation": "straight line joining observed replan endpoints, not actual raw-step path"})
    else:
        result["observed_chunk_endpoint_available"] = False
    return result


def push_conditions(bank, prefix, count):
    import numpy as np
    states = numeric(bank["states"])[prefix:prefix+count+1]
    goal = numeric(bank["goal_state"])
    distances = numeric(bank["step_goal_distance"])
    angle = np.arctan2(np.sin(states[:, 4]-goal[4]), np.cos(states[:, 4]-goal[4]))
    return {"source": "actual cached raw simulator states including chunk start and end",
            "raw_step_realized_path_available": True, "state_path_selection": [prefix, prefix+count+1],
            "xy_units": "simulator canvas coordinate units", "angle_units": "radians",
            "agent_start_xy": states[0, :2].tolist(), "agent_end_xy": states[-1, :2].tolist(),
            "agent_displacement_xy": (states[-1, :2]-states[0, :2]).tolist(),
            "block_start_xy": states[0, 2:4].tolist(), "block_end_xy": states[-1, 2:4].tolist(),
            "block_displacement_xy": (states[-1, 2:4]-states[0, 2:4]).tolist(),
            "block_goal_center_distance_start": float(np.linalg.norm(states[0, 2:4]-goal[2:4])),
            "block_goal_center_distance_end": float(np.linalg.norm(states[-1, 2:4]-goal[2:4])),
            "block_goal_angle_error_start": float(angle[0]), "block_goal_angle_error_end": float(angle[-1]),
            "native_cached_goal_distance_start": float(distances[prefix]),
            "native_cached_goal_distance_end": float(distances[prefix+count]),
            "native_cached_chunk_goal_progress": float(distances[prefix]-distances[prefix+count]),
            "native_cached_cumulative_goal_progress": float(distances[0]-distances[prefix]),
            "contact_or_collision_label_available": False}


def package_bank(root, row):
    import numpy as np
    import torch
    bank = load_development(root, row)
    task = row["task"]
    actions = numeric(bank["actions_raw"])
    expected_dim = 4 if task == "reach_wall" else 2
    if actions.ndim != 2 or actions.shape[1] != expected_dim:
        raise RuntimeError("Raw action dimensions mismatch")
    prefixes = replay_prefixes(bank["replans"], len(actions), task)
    prepared = None
    if task == "pusht":
        source = row["prepared_input"]
        path = root / source["path"]
        if sha(path) != source["sha256"] or bank["input_sha256"] != source["sha256"]:
            raise RuntimeError("Push prepared input hash mismatch")
        prepared = torch.load(path, map_location="cpu", weights_only=False)
        if prepared["split"] != "development" or bank["split"] != "development":
            raise RuntimeError("Push payload split mismatch")
        for key in ("episode", "environment_seed", "planner_seed", "source_trajectory"):
            if prepared[key] != bank[key]:
                raise RuntimeError(f"Prepared/baseline identity mismatch: {key}")
        if len(bank["states"]) != len(actions)+1 or len(bank["observations"]) != len(actions)+1:
            raise RuntimeError("Push initial-plus-postaction state/image layout mismatch")
        if float(bank["baseline_replay_max_state_error"]) > 1e-6 or not bank["baseline_replay_pixels_equal"]:
            raise RuntimeError("Original Push baseline replay did not pass")
    result = []
    for index, (prefix, count, elapsed) in enumerate(prefixes):
        snap = bank["replans"][index]
        planned = numeric(snap["planned_actions_raw"])
        if len(planned) < count or not np.array_equal(planned[:count], actions[prefix:prefix+count]):
            raise RuntimeError("Replan chosen actions disagree with global executed prefix")
        ptr = lambda *keys, **kwargs: pointer(row, "replans", index, *keys, **kwargs)
        item = {"task": task, "episode": row["episode"], "replan": index, "split": "development",
                "environment_seed": row["environment_seed"], "planner_seed": row["planner_seed"],
                "bank_path": row["path"], "bank_sha256": row["sha256"],
                "replay": {"reset_rule": "original seeded initialization then cached raw actions",
                           "warmup_steps_excluded_from_action_prefix": 1 if task == "reach_wall" else 0,
                           "prefix_action_count": prefix, "prefix": pointer(row, "actions_raw", selection=[0, prefix]),
                           "executed_chunk": pointer(row, "actions_raw", selection=[prefix, prefix+count]),
                           "executed_chunk_action_count": count, "elapsed_steps": elapsed,
                           "elapsed_after_chunk": elapsed+count, "model_steps_in_executed_chunk": count/5,
                           "episode_action_progress_fraction": prefix/len(actions),
                           "physics": ptr("simulator"), "planner_rng_before": ptr("planner_rng_before"),
                           "exact_planner_continuation_certified_by_this_package": False,
                           "planner_initialization": "fresh native planner with original seeds and saved pre-CEM RNG" if index == 0 else "warm-start state reconstruction required",
                           "later_replan_planner_requirement": "restore full native CEM warm-start state or deterministically replay prior planning; RNG alone is insufficient"},
                "encoded_visual": ptr("encoded_visual"), "encoded_proprio": ptr("encoded_proprio"),
                "chosen_plan_raw": ptr("planned_actions_raw"), "chosen_plan_normalized": ptr("planned_actions_normalized"),
                "planner_losses": ptr("planner_losses"), "chosen_plan_trace": ptr("chosen_plan_trace"),
                "candidate_costs_available": False,
                "candidate_costs_note": "planner_losses is the saved CEM diagnostic; full candidate costs/identities not cached",
                "action_metrics": motion_summary(actions[prefix:prefix+count], 3 if task == "reach_wall" else 2)}
        if task == "reach_wall":
            item.update({"observation": ptr("observation_visual"), "proprio": ptr("observation_proprio"),
                         "initial_observation": pointer(row, "replans", 0, "observation_visual"),
                         "initial_physics": pointer(row, "replans", 0, "simulator"),
                         "goal": pointer(row, "goal"), "cached_conditions": reach_conditions(bank, index)})
        else:
            item.update({"observation": pointer(row, "observations", index=prefix),
                         "proprio": pointer(row, "proprios", index=prefix),
                         "physical_state": pointer(row, "states", index=prefix),
                         "initial_observation": pointer(row, "observations", index=0),
                         "initial_physics": pointer(row, "initial_state"),
                         "env_info": pointer(row, "env_info"), "goal_state": pointer(row, "goal_state"),
                         "goal_observation": pointer(row["prepared_input"], "expert_observations", index=-1),
                         "goal_proprio": pointer(row["prepared_input"], "expert_proprios", index=-1),
                         "cached_conditions": push_conditions(bank, prefix, count)})
        result.append(item)
    checked = {**row, "tensor_opened_for_packaging": True, "seed_verification": "original bank payload matched declared seeds",
               "development_replans": len(result), "submitted_raw_action_commands": len(actions),
               "prefix_and_chunk_checks_passed": True,
               "upstream_action_replay": "full states and pixels passed during original collection" if task == "pusht" else "all-replan simulator replay not run by this packaging task"}
    return checked, result


def package(args):
    import torch
    torch.set_num_threads(2)
    index = json.loads(args.inventory.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    banks, replans = [], []
    for row in index["banks"]:
        if row["sealed"]:
            banks.append(row)
            continue
        checked, chunk = package_bank(args.project_root, row)
        banks.append(checked); replans.extend(chunk)
        print(json.dumps({"stage": "development_packaged", "task": row["task"], "episode": row["episode"], "replans": len(chunk)}), flush=True)
    counts = {}
    for task in ("reach_wall", "pusht"):
        subset = [row for row in banks if row["task"] == task]
        counts[task] = {"baseline_files": len(subset), "development_episodes": sum(not r["sealed"] for r in subset),
                        "sealed_evaluation_episodes": sum(r["sealed"] for r in subset),
                        "development_replans_packaged": sum(row["task"] == task for row in replans)}
    value = {"schema_version": 1, "complete": True, "scope": "ready index of original baselines and every development replan; no steering performed",
             "project_root": index["project_root"], "pointer_paths": "resolve relative to project_root, not CPU worker root",
             "created_utc": datetime.now(timezone.utc).isoformat(), "coverage": counts,
             "banks": banks, "development_replans": replans, "evaluation_status": "sealed; no evaluation tensors deserialized or labels derived",
             "state_selection": "all development replans in source order; experimental state selection remains to be frozen separately",
             "source_inventory_sha256": sha(args.inventory), "script_sha256": sha(Path(__file__)),
             "source_manifests": index["manifests"], "failures": [],
             "missing_fields": {"both_tasks": ["full initial candidate banks/costs", "complete native CEM warm-start state"],
                                "reach_wall": ["raw-step physical path between replans", "final chunk hand endpoint", "separate initial source-input checksum"],
                                "pusht": ["cached contact/collision labels"]},
             "limitations": ["Packaging checks structure, checksums, seeds, action prefixes and existing replay receipts; it runs no new simulator replay.",
                             "Evaluation bank seeds absent from Push receipts follow original collector rules and await payload verification during authorized frozen evaluation.",
                             "Reach generic initial state predates warmup; actual initial XYZ comes from observation_proprio.",
                             "Sparse Reach waypoint chords and hand-to-goal lines are geometric proxies, not actual raw-step paths or body collision checks.",
                             "Baseline halves and official prepared inputs are not steered or sham outcomes.",
                             "Checkpoint hashes identify existing checkpoint files; original per-worker checkpoint contents were not all independently rehashed."]}
    dump_new(args.output_dir / "ready_index.json", value)
    dump_new(args.output_dir / "DONE.json", {"complete": True, "coverage": counts, "development_replans": len(replans),
        "outputs": [{"path": "ready_index.json", "sha256": sha(args.output_dir / "ready_index.json")}],
        "source_inventory_sha256": sha(args.inventory)})
    print(json.dumps({"stage": "ready_index_complete", "coverage": counts, "replans": len(replans)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    inv = commands.add_parser("inventory")
    for key in ("project-root", "reach-manifest", "prepared-manifest", "checkpoints", "output-dir"):
        inv.add_argument("--"+key, type=Path, required=True)
    inv.add_argument("--reach-dirs", type=Path, nargs="+", required=True)
    inv.add_argument("--pusht-dirs", type=Path, nargs="+", required=True)
    pkg = commands.add_parser("package")
    for key in ("project-root", "inventory", "output-dir"):
        pkg.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    inventory(args) if args.mode == "inventory" else package(args)


if __name__ == "__main__":
    main()
