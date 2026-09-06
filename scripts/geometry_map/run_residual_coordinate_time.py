#!/usr/bin/env python3
"""Discovery-coordinate P3 pulse diagnostics: fixed actions, then short CEM forks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import sys
import numpy as np
import torch
from capture_horizon_coordinates import CaptureP3Horizons, exact, decode_physical, same_physics, reconstruct_fresh

TIMES = (1, 3, 6)
RANKS = (1, 2, 8)
PHYSICAL_STARTS = (0, 4, 10)


def native_unroll_actions(args, kwargs):
    actions = kwargs.get("act_suffix", args[1] if len(args) > 1 else None)
    if actions is None:
        raise RuntimeError("Native planner unroll missing action sequence")
    return actions


def hand_goal_metrics(start, states):
    states = np.asarray(states)
    goal = states[0, -3:]
    if not np.array_equal(states[:, -3:], np.repeat(goal[None], len(states), axis=0)):
        raise RuntimeError("Physical task goal changed during short fork")
    distances = np.linalg.norm(states[:, :3]-goal, axis=1)
    initial = float(np.linalg.norm(np.asarray(start)-goal))
    return {"initial_hand_goal_distance_m": initial, "final_hand_goal_distance_m": float(distances[-1]),
            "minimum_hand_goal_distance_m": float(min(initial, distances.min())),
            "hand_goal_progress_m": initial-float(distances[-1])}


def summarize_saved(args):
    """One-thread CPU readback only; no new model/simulator call or fitting."""
    from package_steering_banks import sha, load_development
    from protocol import write_json_atomic
    done = json.loads((args.directory/"DONE.json").read_text())
    entries = {row["path"]: row for row in done["outputs"]}
    if not done["complete"]:
        raise RuntimeError("Incomplete physical run cannot be summarized as complete")
    protocol = json.loads((args.directory/"protocol.json").read_text())
    if sha(args.directory/"protocol.json") != entries["protocol.json"]["sha256"]:
        raise RuntimeError("Source protocol checksum mismatch")
    inventory = json.loads(args.inventory.read_text())
    source_rows = [row for row in inventory["banks"] if row["task"] == "reach_wall" and row["episode"] in PHYSICAL_STARTS]
    readout = dict(np.load(args.readout, allow_pickle=False))
    rows = []
    for episode in PHYSICAL_STARTS:
        name = f"episode-{episode:03d}-planner-unsteered.pt"
        if name not in entries:
            continue
        source = next(row for row in source_rows if row["episode"] == episode)
        bank = load_development(args.project_root, source)
        start = bank["replans"][0]["observation_proprio"].reshape(-1)[:3].numpy()
        def load(arm):
            path = args.directory/f"episode-{episode:03d}-planner-{arm}.pt"
            if sha(path) != entries[path.name]["sha256"]:
                raise RuntimeError("Physical source checksum mismatch before tensor access")
            return torch.load(path, map_location="cpu", weights_only=False)
        baseline = load("unsteered")
        baseline_metrics = hand_goal_metrics(start, baseline["fork"]["states"])
        baseline_common = decode_physical(baseline["first_candidates"]["pooled_visual"].numpy(), readout)
        baseline_selected = np.asarray(baseline["metrics"]["own_selected_plan_decoded_xyz"])
        for arm in ("unsteered", "positive", "positive_sham", "negative", "negative_sham"):
            value = baseline if arm == "unsteered" else load(arm)
            metrics = hand_goal_metrics(start, value["fork"]["states"])
            common = decode_physical(value["first_candidates"]["pooled_visual"].numpy(), readout)
            selected = np.asarray(value["metrics"]["own_selected_plan_decoded_xyz"])
            commands = value["fork"]["raw_commands"].numpy()
            base_commands = baseline["fork"]["raw_commands"].numpy()
            actual_mocap = np.asarray(value["fork"]["controls"][-1]["mocap_position"])[0]
            base_mocap = np.asarray(baseline["fork"]["controls"][-1]["mocap_position"])[0]
            rewards = value["fork"]["rewards"]
            # Native Reach-v2 reward is10*tolerance(tcp_distance, bounds<=.05).
            # Every unsaturated reward excludes native success. Do not infer
            # positive success from rounded saturation or a hand-point proxy.
            success_excluded = max(rewards) < 10.
            rows.append({**value["metrics"], **metrics,
                         "hand_goal_progress_change_vs_baseline_m": metrics["hand_goal_progress_m"]-baseline_metrics["hand_goal_progress_m"],
                         "common_first300_predicted_z_shift_h3_m": float((common[2, :, 2]-baseline_common[2, :, 2]).mean()),
                         "common_first300_predicted_z_shift_h6_m": float((common[5, :, 2]-baseline_common[5, :, 2]).mean()),
                         "own_selected_plan_predicted_z_shift_h3_m": float(selected[2, 2]-baseline_selected[2, 2]),
                         "own_plan_first3_physical_readout_error_m": np.linalg.norm(selected[:3]-value["fork"]["states"][4::5, :3], axis=1).tolist(),
                         "raw_command_z_sum_change": float(commands[:, 2].sum()-base_commands[:, 2].sum()),
                         "actual_final_mocap_xyz_change": (actual_mocap-base_mocap).tolist(),
                         "native_reward_sum": float(sum(rewards)), "native_reward_max": float(max(rewards)),
                         "native_ever_success": False if success_excluded else None,
                         "native_success_provenance": "excluded by all recorded nativev2 rewards<10" if success_excluded else "missing recorded info.success; saturatedreward alone not treated as exact positive success",
                         "native_min_tcp_goal_distance_m": None,
                         "native_tcp_note": "TCP differs from observed hand; no handthreshold substituted, no new kinematic/simulator pass"})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = args.output_dir/"physical_effects.json"
    write_json_atomic(report, {"complete": True, "scope": "CPU-only saved15-step development physical arms; no new forwards/rollouts",
                              "source_done_sha256": sha(args.directory/"DONE.json"), "candidate_sha256": protocol["candidate_sha256"],
                              "script_sha256": sha(Path(__file__)), "rows": rows,
                              "native_success_source": "metaworld/envs/sawyer_reach_wall_v3.py:evaluate_state and compute_reward v2; success<=.05 TCPdistance implies reward10",
                              "interpretation": "Common300 shifts precede optimization; selected-plan forecasts are arm-own actions, not fixed-action contrasts; hand-goal progress is physical but not nativeTCP success"})
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": [{"path": report.name, "sha256": sha(report), "bytes": report.stat().st_size}]})
    print(json.dumps({"event": "physical_effects_summary_complete", "rows": rows, "sha256": sha(report)}), flush=True)


def summary_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    for name in ("directory", "project-root", "inventory", "readout", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    summarize_saved(parser.parse_args())


def aggregate_fixed_rows(rows):
    """Equal-episode descriptive means; six horizons are not six episodes."""
    keys = ("requested_rank", "intervention_step", "sign", "sham", "measurement_horizon")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        episodes = [r["episode"] for r in values]
        if len(set(episodes)) != len(episodes):
            raise RuntimeError("Duplicate episode within fixed-condition aggregate")
        result.append({**dict(zip(keys, key)), "episodes": sorted(episodes), "episode_count": len(episodes),
                       "mean_decoded_z_shift_m": float(np.mean([r["decoded_xyz_shift"][2] for r in values])),
                       "mean_latent_error_change": float(np.mean([r["latent_error_change"] for r in values])),
                       "mean_physical_readout_error_m": float(np.mean([r["physical_readout_error_m"] for r in values])),
                       "latent_error_improved_count": sum(r["latent_error_change"] < 0 for r in values)})
    return result


def aggregate_main():
    """Join already verified JSONs only; no tensor access, fitting or simulation."""
    from package_steering_banks import sha
    from protocol import write_json_atomic
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-local", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fixed, physical, sources = [], [], []
    for worker, directory in (("49766237", "results-v1"), ("49155754", "results-v2"), ("49902461", "results-v2")):
        base = args.aggregate_local/("worker-"+worker)
        for root, pattern, target in ((base/directory, "episode-*-fixed.json", fixed), (base/"summary-v1", "physical_effects.json", physical)):
            receipt_path = root/("REMOTE_COPY_RECEIPT.json" if worker == "49766237" and root.name == "results-v1" else "DONE.json")
            receipt = json.loads(receipt_path.read_text())
            entries = {r["path"]: r for r in receipt["outputs"]}
            for path in sorted(root.glob(pattern)):
                if sha(path) != entries[path.name]["sha256"]:
                    raise RuntimeError("JSON source hash mismatch")
                value = json.loads(path.read_text())
                if not value["complete"]:
                    raise RuntimeError("Incomplete JSON source")
                target.extend(value["rows"])
                sources.append({"path": str(path), "sha256": sha(path), "receipt_sha256": sha(receipt_path)})
    if len(fixed) != 2592 or sorted(set(r["episode"] for r in fixed)) != list(range(12)) or len(physical) != 15:
        raise RuntimeError("Expected432 fixed arms/six horizons and15 physical arms")
    comparisons = []
    for arm in ("positive", "negative"):
        semantic = [r for r in physical if r["arm"] == arm]
        paired = []
        for row in semantic:
            sham = next(r for r in physical if r["episode"] == row["episode"] and r["arm"] == arm+"_sham")
            paired.append({"episode": row["episode"], "progress_vs_baseline_m": row["hand_goal_progress_change_vs_baseline_m"],
                           "progress_vs_matched_sham_m": row["hand_goal_progress_m"]-sham["hand_goal_progress_m"],
                           "same_first300_predicted_z_shift_h3_m": row["common_first300_predicted_z_shift_h3_m"],
                           "own_plan_predicted_z_shift_h3_m": row["own_selected_plan_predicted_z_shift_h3_m"],
                           "raw_command_z_sum_change": row["raw_command_z_sum_change"],
                           "actual_hand_z_shift_m": row["physical_end_xyz_shift"][2]})
        comparisons.append({"arm": arm, "episode_count": len(paired), "paired_rows": paired,
                            "mean_progress_vs_baseline_m": float(np.mean([r["progress_vs_baseline_m"] for r in paired])),
                            "mean_progress_vs_matched_sham_m": float(np.mean([r["progress_vs_matched_sham_m"] for r in paired])),
                            "better_than_sham_count": sum(r["progress_vs_matched_sham_m"] > 0 for r in paired),
                            "initial_decoded_z_vs_physical_z_opposition_count": sum(r["same_first300_predicted_z_shift_h3_m"]*r["actual_hand_z_shift_m"] < 0 for r in paired)})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"complete": True, "sources": sources, "script_sha256": sha(Path(__file__)),
              "candidate_sha256": "07325ea079345e65d961cb8c598955ade195de51f1a8e9b9c85a1236b4cf9543",
              "fixed_arm_count": 432, "fixed_episode_count": 12, "fixed_horizon_row_count": 2592,
              "physical_arm_count": 15, "physical_episode_count": 3, "physical_comparisons": comparisons,
              "fixed_condition_profiles": aggregate_fixed_rows(fixed), "physical_rows": physical,
              "development_operator_decision": "Neither current signed rank8/pulse1 operator is promoted to a full held study; positive is worse than its matched sham on all3 starts, negative is mixed/nearly neutral",
              "limitations": "Descriptive predeclared development results; no significance or confirmation claim. Same-first300 forecasts precede optimization, while selected-plan forecasts use each arm's own actions; opposition is consistent with planner compensation, not proof. Hand-goal progress is not native TCP success; no new model/simulator calls."}
    path = args.output_dir/"residual_coordinate_results.json"
    write_json_atomic(path, result)
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": [{"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size}]})
    print(json.dumps({"event": "residual_aggregate_complete", "sha256": sha(path), "physical_comparisons": comparisons}), flush=True)


def projected_difference(current, donor, basis):
    return ((donor-current) @ basis) @ basis.T


def cap_delta(delta, radius):
    if radius <= 0:
        raise ValueError("Positive frozen discovery cap required")
    return delta * (radius/delta.norm(dim=-1).clamp_min(torch.finfo(delta.dtype).tiny)).clamp(max=1)[:, None]


def match_norm(direction, reference):
    norm, desired = direction.norm(dim=-1), reference.norm(dim=-1)
    if ((norm == 0) & (desired > 0)).any():
        raise RuntimeError("Zero sham direction cannot match nonzero semantic norm")
    return direction * (desired/norm.clamp_min(torch.finfo(direction.dtype).tiny))[:, None]


def random_basis(dim, rank, seed=2026090503):
    generator = torch.Generator().manual_seed(seed+rank)
    basis = torch.linalg.qr(torch.randn(dim, rank, generator=generator, dtype=torch.float64)).Q
    return basis


class DiscoveryDonors:
    def __init__(self, artifact, rank, device="cpu"):
        self.basis = torch.as_tensor(artifact[f"basis_raw_rank{rank}"], dtype=torch.float32, device=device)
        self.donors = torch.as_tensor(artifact["donor_p3"], dtype=torch.float32, device=device)
        self.coef = torch.as_tensor(artifact["coordinate_coef_raw"], dtype=torch.float32, device=device)[2]
        self.intercept = torch.as_tensor(artifact["coordinate_intercept_raw"], dtype=torch.float32, device=device)[2]
        self.radius, self.rank = float(artifact["cap_radius"]), self.basis.shape[1]
        if self.basis.shape[0] != 400 or not 0 < self.rank <= rank:
            raise ValueError("Malformed raw P3 basis")
        if not torch.allclose(self.basis.T@self.basis, torch.eye(self.rank, device=device), atol=1e-5):
            raise ValueError("P3 basis is not raw-Euclidean orthonormal")
        if not all(torch.isfinite(x).all() for x in (self.basis, self.donors, self.coef, self.intercept)):
            raise ValueError("Nonfinite frozen candidate")
        self.sham_basis = random_basis(400, self.rank).to(self.basis)
        self.complement = self.donors-(self.donors@self.basis)@self.basis.T
        self.donor_scores = self.donors@self.coef+self.intercept

    def delta(self, current, sign, sham=False):
        if sign not in (-1, 1):
            raise ValueError("Signed donor selection requires+1 or-1")
        complement = current-(current@self.basis)@self.basis.T
        distance = (complement.square().sum(1)[:, None]+self.complement.square().sum(1)[None]
                    -2*complement@self.complement.T).clamp_min(0)
        scores = current@self.coef+self.intercept
        eligible = sign*(self.donor_scores[None]-scores[:, None]) > 0
        available = eligible.any(1)
        index = distance.masked_fill(~eligible, torch.inf).argmin(1)
        donor = self.donors[index]
        donor = torch.where(available[:, None], donor, current)
        semantic = cap_delta(projected_difference(current, donor, self.basis), self.radius)
        requested = match_norm(projected_difference(current, donor, self.sham_basis), semantic) if sham else semantic
        return requested, {"donor_index": torch.where(available, index, -torch.ones_like(index)),
                           "donor_available": available, "recipient_score": scores,
                           "donor_score": torch.where(available, self.donor_scores[index], scores),
                           "complement_distance_squared": distance.gather(1, index[:, None])[:, 0],
                           "semantic_norm": semantic.norm(dim=-1), "requested_norm": requested.norm(dim=-1)}


class ResidualPulse:
    def __init__(self, block, native_unroll, donors, imagined_step, sign=1, sham=False, identity=False):
        if imagined_step not in TIMES:
            raise ValueError("Unfrozen pulse time")
        self.block, self.native_unroll, self.donors = block, native_unroll, donors
        self.imagined_step, self.sign, self.sham, self.identity = imagined_step, sign, sham, identity
        self.step, self.records = 0, []

    def hook(self, _module, _inputs, output):
        self.step += 1
        if self.step != self.imagined_step:
            return output
        if output.ndim != 3 or output.shape[-1] != 400 or output.shape[1] % 256:
            raise RuntimeError("Expected native P3 residual [B,T*256,400]")
        if self.identity:
            return output
        current = output[:, -256:].float().mean(1)
        delta, metadata = self.donors.delta(current, self.sign, self.sham)
        edited = output.clone()
        edited[:, -256:] = output[:, -256:]+delta.to(output)[:, None]
        actual = edited[:, -256:].float().mean(1)-current
        basis = self.donors.sham_basis if self.sham else self.donors.basis
        self.records.append({**{key: value.detach().cpu() for key, value in metadata.items()},
                             "before_pooled": current.detach().cpu(), "requested_delta": delta.detach().cpu(),
                             "actual_rounded_delta": actual.detach().cpu(), "actual_rounded_norm": actual.norm(dim=-1).cpu(),
                             "requested_orthogonal_error": (delta-(delta@basis)@basis.T).norm(dim=-1).cpu()})
        return edited

    def unroll(self, *args, **kwargs):
        self.step = 0
        result = self.native_unroll(*args, **kwargs)
        if self.step != 6:
            raise RuntimeError(f"Pulse requires exactH6 unroll, observed{self.step}")
        return result

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()


def load_reference(directories, episode):
    from package_steering_banks import sha
    if not 0 <= episode < 12:
        raise RuntimeError("Held episode refused before tensor access")
    matches = [path/f"episode-{episode:03d}.pt" for path in directories if (path/f"episode-{episode:03d}.DONE.json").exists()]
    if len(matches) != 1:
        raise RuntimeError("Exactly one completed development horizon reference required")
    path = matches[0]
    done = json.loads(path.with_suffix(".DONE.json").read_text())
    entry = next(row for row in done["outputs"] if row["path"] == path.name)
    if not done["complete"] or entry["episode"] != episode or sha(path) != entry["sha256"]:
        raise RuntimeError("Horizon reference checksum/episode mismatch")
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not result["fresh30action_physics_and_pixels_repeat_exact"]:
        raise RuntimeError("Missing exact-action physical truth")
    return result, sha(path)


def arm_rows(prediction, p3, baseline, truth, readout, episode, pulse, rank, sign, sham, frame):
    visual = prediction["visual"][1:, 0].float().cpu().reshape(6, -1, 384)
    base = baseline["visual"][1:, 0].float().cpu().reshape_as(visual)
    actual = torch.cat(truth["encoded_visual"], 0).reshape_as(visual)
    decoded = decode_physical(visual.mean(1), readout)
    base_decoded = decode_physical(base.mean(1), readout)
    true_xyz = truth["states"][5::5, :3]
    error, base_error = (visual-actual).square().mean((1, 2)), (base-actual).square().mean((1, 2))
    return [{"episode": episode, "intervention_step": pulse, "measurement_horizon": h+1,
             "requested_rank": rank, "sign": sign, "sham": sham, "coordinate_frame": frame,
             "predicted_xyz": decoded[h].tolist(), "truth_xyz": true_xyz[h].tolist(),
             "decoded_xyz_shift": (decoded[h]-base_decoded[h]).tolist(),
             "physical_readout_error_m": float(np.linalg.norm(decoded[h]-true_xyz[h])),
             "latent_mse_to_actual": float(error[h]), "baseline_latent_mse_to_actual": float(base_error[h]),
             "latent_error_change": float(error[h]-base_error[h]),
             "p3_pooled_raw_norm": float(p3[h].mean(1).norm(dim=-1)[0])} for h in range(6)]


@torch.no_grad()
def fixed_actions(args, episode, wm, preprocessor, cfg, artifact, readout, deadline):
    from package_steering_banks import load_development, sha
    from causal_planner_forks import reconstruct, close_env
    from protocol import write_json_atomic
    reference, reference_sha = load_reference(args.reference_dirs, episode)
    row = next(row for row in args.source_rows if row["episode"] == episode)
    bank = load_development(args.project_root, row)
    env, agent, z, checks = reconstruct_fresh(cfg, wm, preprocessor, bank)
    expected_goal = reference["initial_checks"].get("fresh_goal_sha256")
    if expected_goal and expected_goal != checks["fresh_goal_sha256"]:
        raise RuntimeError("Fresh fixed-action goal differs from physical reference")
    try:
        actions = reference["normalized_actions"].cuda()[:, None]
        baseline = wm.unroll(z.clone(), act_suffix=actions)
        baseline_context = {key: z[key].detach().cpu().clone() for key in z.keys()}
        results, outputs = [], []
        block = wm.model.predictor.predictor_blocks[3]
        donors_by_rank = {rank: DiscoveryDonors(artifact, rank, "cuda") for rank in RANKS}
        with ResidualPulse(block, wm.unroll, donors_by_rank[8], 1, identity=True) as identity:
            unchanged = identity.unroll(z.clone(), act_suffix=actions)
        for key in ("visual", "proprio"):
            exact(unchanged[key], baseline[key], "fullH6 zero/same-state identity " + key)
        for rank in RANKS:
            for pulse in TIMES:
                for sign in (1, -1):
                    semantic_norms = None
                    for sham in (False, True):
                        if time.monotonic() > deadline:
                            raise RuntimeError("Frozen per-worker1800s budget reached before next fixed-action arm")
                        name = f"episode-{episode:03d}-rank{rank}-j{pulse}-{'positive' if sign > 0 else 'negative'}-{'sham' if sham else 'semantic'}"
                        with ResidualPulse(block, wm.unroll, donors_by_rank[rank], pulse, sign, sham) as patch:
                            with CaptureP3Horizons(block) as capture:
                                prediction = patch.unroll(z.clone(), act_suffix=actions)
                        for key in ("visual", "proprio"):
                            exact(prediction[key][:pulse], baseline[key][:pulse], "no pre-pulse effect " + key)
                        for key in baseline_context:
                            exact(z[key], baseline_context[key], "immutable model context " + key)
                        record = patch.records[0]
                        if not sham:
                            semantic_norms = record["requested_norm"]
                        elif not torch.allclose(record["requested_norm"], semantic_norms, atol=1e-6, rtol=1e-6):
                            raise RuntimeError("Requested sham/semantic raw norm mismatch")
                        p3 = torch.stack(capture.values)
                        rows = arm_rows(prediction, p3, baseline, reference["truth"], readout, episode, pulse, rank, sign, sham, str(artifact["target_name"]))
                        file = args.output_dir/(name+".pt")
                        torch.save({"episode": episode, "rows": rows, "reference_sha256": reference_sha,
                                    "candidate_sha256": args.candidate_sha256, "record": record,
                                    "actual_rank": donors_by_rank[rank].rank, "p3_pooled": p3.mean(2),
                                    "prediction": {key: prediction[key].detach().cpu() for key in ("visual", "proprio")}}, file)
                        outputs.append({"path": file.name, "sha256": sha(file), "bytes": file.stat().st_size})
                        results.extend(rows)
                        print(json.dumps({"event": "residual_fixed_arm_complete", "arm": name,
                                          "actual_rank": donors_by_rank[rank].rank, "donor_available": record["donor_available"].tolist()}), flush=True)
        report = args.output_dir/f"episode-{episode:03d}-fixed.json"
        write_json_atomic(report, {"complete": True, "episode": episode, "reference_sha256": reference_sha,
                                  "candidate_sha256": args.candidate_sha256, "identity_exact": True, "rows": results})
        outputs.append({"path": report.name, "sha256": sha(report), "bytes": report.stat().st_size})
        return outputs
    finally:
        close_env(env)


def physical15(env, actions):
    from collect_on_policy_bank import physics_snapshot
    from precompute_native_replay import ReachControlCapture
    base = env.proprio_env.unwrapped
    states, frames, controls, physics, rewards = [], [], [], [], []
    with ReachControlCapture(base) as capture:
        for action in actions:
            capture.records.clear()
            obs, reward, done, infos = env.step_multiple(action[None].cpu())
            if len(obs) != 1:
                raise RuntimeError("Short native physical fork ended early")
            states.append(np.asarray(infos[0]["state"]).copy())
            frames.append(obs[0].cpu())
            controls.append({"xyz_setter": list(capture.records), "actuator_ctrl": np.asarray(base.data.ctrl).copy(),
                             "mocap_position": np.asarray(base.data.mocap_pos).copy()})
            physics.append(physics_snapshot(env)); rewards.append(float(reward[0]))
    return {"states": np.stack(states), "frames": torch.stack(frames), "controls": controls,
            "physics": physics, "raw_commands": actions.cpu(), "rewards": rewards}


@torch.no_grad()
def planner_physics(args, episode, wm, preprocessor, cfg, artifact, readout, deadline):
    from package_steering_banks import load_development, sha
    from causal_planner_forks import reconstruct, close_env, visual_pooled
    from collect_on_policy_bank import physics_snapshot
    from protocol import write_json_atomic
    row = next(row for row in args.source_rows if row["episode"] == episode)
    bank = load_development(args.project_root, row)
    donors = DiscoveryDonors(artifact, 8, "cuda")
    output_rows, outputs, baseline = [], [], None
    for arm, sign, sham in (("unsteered", 1, False), ("positive", 1, False), ("positive_sham", 1, True),
                            ("negative", -1, False), ("negative_sham", -1, True)):
        if time.monotonic() > deadline:
            raise RuntimeError("Frozen per-worker1800s budget reached before next CEM arm")
        env, agent, z, checks = reconstruct_fresh(cfg, wm, preprocessor, bank)
        initial = physics_snapshot(env)
        if baseline is not None:
            if checks["fresh_goal_sha256"] != baseline["fresh_goal_sha256"]:
                raise RuntimeError("Fresh planning goal changed across arms")
            same_physics(initial, baseline["initial_physics"], "paired physical start")
            for key in ("visual", "proprio"):
                exact(z[key], baseline["encoded_context"][key], "paired planning context " + key)
        native = agent.planner.unroll
        patch = ResidualPulse(wm.model.predictor.predictor_blocks[3], native, donors, 1, sign, sham)
        first = {}
        def tracked(*a, **kw):
            prediction = native(*a, **kw) if arm == "unsteered" else patch.unroll(*a, **kw)
            actions = native_unroll_actions(a, kw)
            if actions.shape[1] == 300 and not first:
                first.update({"actions": actions.cpu().clone(), "costs": agent.objective(prediction, actions).cpu(),
                              "pooled_visual": visual_pooled(prediction, actions).cpu(), "proprio": prediction["proprio"].cpu()})
            return prediction
        agent.planner.unroll = tracked
        try:
            print(json.dumps({"event": "residual_native_cem_started", "episode": episode, "arm": arm}), flush=True)
            if arm == "unsteered":
                prefix = agent.plan(z.clone(), steps_left=bank["replans"][0]["steps_left_model"])
            else:
                with patch:
                    prefix = agent.plan(z.clone(), steps_left=bank["replans"][0]["steps_left_model"])
            mean = agent.planner._prev_mean.detach().cpu()
            if baseline is not None:
                exact(first["actions"], baseline["first_candidates"]["actions"], "paired first300 candidates")
            raw = preprocessor.denormalize_actions(prefix.cpu().reshape(15, 4))
            result = {"episode": episode, "arm": arm, "initial_physics": initial, "fresh_goal_sha256": checks["fresh_goal_sha256"],
                      "encoded_context": {key: z[key].cpu() for key in ("visual", "proprio")},
                      "selected_full_mean": mean, "selected_prefix": prefix.cpu(), "first_candidates": first,
                      "selected_prediction": agent._predicted_best_encs_over_iterations[-1].detach().cpu(),
                      "fork": physical15(env, raw), "edit_records": patch.records,
                      "candidate_sha256": args.candidate_sha256, "actual_rank": donors.rank}
            if baseline is None:
                baseline = result
            selected = result["selected_prediction"]["visual"][1:].float().reshape(6, -1, 384).mean(1)
            decoded = decode_physical(selected, readout)
            metrics = {"episode": episode, "arm": arm, "intervention_step": 1, "actual_rank": donors.rank,
                       "first_candidate_argmin": int(first["costs"].argmin()),
                       "first_candidate_rank_changes": int((first["costs"].argsort().argsort()!=baseline["first_candidates"]["costs"].argsort().argsort()).sum()),
                       "raw_command_z_sum": float(raw[:, 2].sum()),
                       "raw_command_l2": float(raw.norm()), "physical_end_xyz": result["fork"]["states"][-1, :3].tolist(),
                       "physical_end_xyz_shift": (result["fork"]["states"][-1, :3]-baseline["fork"]["states"][-1, :3]).tolist(),
                       "own_selected_plan_decoded_xyz": decoded.tolist(),
                       "own_plan_prediction_semantics": "Each arm optimizes its own plan; not a fixed-action comparison"}
            result["metrics"] = metrics
            file = args.output_dir/f"episode-{episode:03d}-planner-{arm}.pt"
            torch.save(result, file)
            outputs.append({"path": file.name, "sha256": sha(file), "bytes": file.stat().st_size})
            output_rows.append(metrics)
            print(json.dumps({"event": "residual_physical_arm_complete", "episode": episode, "arm": arm}), flush=True)
        finally:
            agent.planner.unroll = native
            close_env(env)
    report = args.output_dir/f"episode-{episode:03d}-planner.json"
    write_json_atomic(report, {"complete": True, "rows": output_rows, "scope": "three predeclared development starts, short15-step physical forks; no efficacy confirmation"})
    outputs.append({"path": report.name, "sha256": sha(report), "bytes": report.stat().st_size})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "project-root", "inventory", "candidate", "readout", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--reference-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--stage", choices=("fixed", "planner", "both"), default="both")
    args = parser.parse_args()
    from package_steering_banks import sha
    from precompute_native_replay import safe_rows
    from protocol import write_json_atomic
    from causal_planner_forks import setup_cfg
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    if sha(args.candidate) != args.candidate_sha256:
        raise RuntimeError("Frozen discovery candidate checksum mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.source_rows = safe_rows(json.loads(args.inventory.read_text()), "reach_wall", args.episodes)
    artifact, readout = dict(np.load(args.candidate, allow_pickle=False)), dict(np.load(args.readout, allow_pickle=False))
    protocol = {"candidate_sha256": args.candidate_sha256, "script_sha256": sha(Path(__file__)),
                "target_frame": str(artifact["target_name"]), "target_index": 2, "beta": 1., "cap_radius": float(artifact["cap_radius"]),
                "cap_order": "beta1 projected donor difference, then raw Euclidean radial cap; sham matched AFTER cap",
                "donor_policy": "nearest raw orthogonal-complement discovery state with frozen linear coordinate2 above/below recipient; tie:first frozen donor index",
                "missing_donor_policy": "identity/no edit if no eligible donor; record index-1/availablefalse, never extrapolate or use futuretruth",
                "signed_semantics": "positive/negative select distinct above/below discovery donors, not symmetric doses of one vector",
                "sham": "same donor/current difference projected into fixed seed2026090503+actualrank random orthonormal subspace, rescaled to semantic rawL2",
                "ranks": list(RANKS), "times": list(TIMES), "episodes": args.episodes, "physical_starts": list(PHYSICAL_STARTS),
                "stage": args.stage, "physical_arms": ["unsteered", "positive", "positive_sham", "negative", "negative_sham"],
                "identity": "fullH6 fixed-action zero/same-state forward bitexact, no duplicate physical identity episode",
                "goal_encoding_policy": "original raw goal exact; fresh fullprecision goal hash identical across newpairedarms; oldfloat16cache discrepancy descriptive only",
                "physical_intervention": "available rank8 pulse1 only, nativeH6/300/15 CEM then15rawsteps",
                "per_worker_budget_seconds": 1800, "no_held_episodes": True, "no_confirmation_or_efficacy_claim": True,
                "readout_sha256": sha(args.readout), "inventory_sha256": sha(args.inventory)}
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    deadline = time.monotonic()+1800
    try:
        wm, preprocessor, provenance = load_headless_metaworld(args.repo)
        wm.eval().requires_grad_(False)
        if any(row["model"]["checkpoint_sha256"] != sha(Path(provenance["checkpoint"])) for row in args.source_rows):
            raise RuntimeError("Frozen checkpoint differs from development references")
        before = parameters_sha(wm)
        cfg = setup_cfg(args.config, args.output_dir, wm)
        outputs = []
        if args.stage in ("fixed", "both"):
            for episode in args.episodes:
                outputs += fixed_actions(args, episode, wm, preprocessor, cfg, artifact, readout, deadline)
        if args.stage in ("planner", "both"):
            for episode in args.episodes:
                if episode in PHYSICAL_STARTS:
                    outputs += planner_physics(args, episode, wm, preprocessor, cfg, artifact, readout, deadline)
        if parameters_sha(wm) != before:
            raise RuntimeError("Frozen weights changed")
        report = args.output_dir/"residual_coordinate_time.json"
        write_json_atomic(report, {**protocol, "complete": True, "model": provenance, "checkpoint_sha256": sha(Path(provenance["checkpoint"])),
                                  "parameters_unchanged": True, "parameters_sha256": before, "outputs": outputs})
        outputs += [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json")]
        write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})
    except Exception as exc:
        write_json_atomic(args.output_dir/"FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    aggregate_main() if "--aggregate-local" in sys.argv else (summary_main() if "--summarize" in sys.argv else main())
