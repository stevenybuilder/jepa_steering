#!/usr/bin/env python3
"""Development-only residual search; every candidate has a paired norm sham."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from run_residual_coordinate_time import DiscoveryDonors, load_reference, native_unroll_actions
from capture_horizon_coordinates import exact, decode_physical
from protocol import file_sha256, write_json_atomic


def token_mask(name, device="cpu"):
    grid = torch.arange(256, device=device).reshape(16, 16)
    masks = {"all": torch.ones_like(grid, dtype=torch.bool), "top": grid//16 < 8, "bottom": grid//16 >= 8,
             "left": grid%16 < 8, "right": grid%16 >= 8, "center": ((grid//16 >= 4)&(grid//16 < 12)&(grid%16 >= 4)&(grid%16 < 12)),
             "checker": (grid//16+grid%16)%2 == 0}
    if name not in masks:
        raise ValueError("Unknown predetermined token mask")
    return masks[name].reshape(256)


def masked_equal_energy(delta, mask):
    desired = delta.flatten(1).norm(dim=-1)
    result = delta*mask[None, :, None]
    available = result.flatten(1).norm(dim=-1)
    if ((available == 0)&(desired > 0)).any():
        raise RuntimeError("Nonzero edit has no direction within selected token mask")
    return result*(desired/available.clamp_min(1e-30))[:, None, None]


def signed_permutation_sham(delta, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(delta.shape[-1], generator=generator).to(delta.device)
    signs = (torch.randint(0, 2, (delta.shape[-1],), generator=generator)*2-1).to(delta)
    return delta[..., permutation]*signs


def pilot_candidates():
    result = []
    for mask in ("all", "top", "bottom"):
        for pulse in (1, 3):
            for dose in (.25, 1.):
                for sign in (-1, 1):
                    result.append({"id": f"pilot-{len(result):03d}", "family": "coordinate", "block": 3,
                                   "site": "residual", "head": None, "mask": mask, "rank": 8,
                                   "pulse": pulse, "dose": dose, "sign": sign})
    return result


def broad_candidates(seed=2026090611):
    rng = np.random.default_rng(seed)
    result = []
    def add(**changes):
        row = {"family": "branch_scale", "block": 0, "site": "residual", "head": None, "mask": "all", "rank": 1,
               "pulse": 1, "dose": .25, "sign": -1}
        row.update(changes); row["id"] = f"broad0-{len(result):03d}"
        result.append(row)
    for block in range(6):
        for head in range(16):
            add(block=block, site="attention_preproj", head=head, pulse=1+(block+head)%6,
                dose=(.125, .25, .5, 1.)[head%4], sign=(-1, 1)[(head//4+block)%2])
    for block in range(6):
        for site in ("residual", "attention_preproj", "mlp_output"):
            for family in ("action_gain", "pca_gain"):
                for rank in (1, 4, 8, 16):
                    add(block=block, site=site, family=family, rank=rank, pulse=int(rng.integers(0, 7)),
                        mask=str(rng.choice(["all", "top", "bottom", "left", "right", "center", "checker"])),
                        dose=float(rng.choice([.125, .25, .5, 1.])), sign=int(rng.choice([-1, 1])))
    while len(result) < 256:
        add(block=int(rng.integers(0, 6)), site=str(rng.choice(["residual", "mlp_output"])), pulse=int(rng.integers(0, 7)),
            mask=str(rng.choice(["all", "top", "bottom", "left", "right", "center", "checker"])),
            dose=float(rng.choice([.125, .25, .5, 1.])), sign=int(rng.choice([-1, 1])))
    return result


class BroadPatch:
    def __init__(self, wm, candidate, bank, sham=False, identity=False):
        from residual_search_hooks import ComponentPatch
        self.wm, self.candidate, self.bank = wm, candidate, bank
        self.native = wm.unroll
        self.name = str(candidate["block"])+"/"+candidate["site"]
        self.component = ComponentPatch(wm.model.predictor.predictor_blocks[candidate["block"]], candidate, self.delta,
                                        float(bank[str(candidate["block"])+"/cap_radius"]), sham=sham, identity=identity)
        self.records = []
        self.norm_targets = None
        self.actual_rank = None

    def delta(self, current, step):
        c = self.candidate
        if c["family"] == "branch_scale":
            return current
        basis = self.bank[self.name+("/action_basis" if c["family"] == "action_gain" else "/pca_basis")]
        rank = min(c["rank"], len(basis)); self.actual_rank = rank
        basis = basis[:rank]
        if rank == 0:
            raise RuntimeError("No identifiable discovery subspace")
        if c["family"] == "action_gain":
            actions = self.raw[step-1, :, :, :3].reshape(len(current), 15)
            values = actions@self.bank[self.name+"/jacobian"].reshape(15, -1)
        elif c["family"] == "pca_gain":
            values = (current-self.bank[self.name+"/mean"]).flatten(1)
        else:
            raise RuntimeError("Unknown broad candidate family")
        return ((values@basis.T)@basis).reshape_as(current)

    def unroll(self, *args, **kwargs):
        actions = native_unroll_actions(args, kwargs)
        # Released normalization statistics live onCPU. This is a read-only
        # command-coordinate conversion, not a modification of model actions.
        if self.candidate["family"] == "action_gain":
            self.raw = self.wm.preprocessor.denormalize_actions(actions.reshape(-1, 4).cpu()).to(actions).reshape(*actions.shape[:2], 5, 4)
        self.component.norm_targets = self.norm_targets
        self.component.reset()
        before = len(self.component.records)
        result = self.native(*args, **kwargs)
        if self.component.step != 6:
            raise RuntimeError("Broad patch expects full nativeH6")
        self.records.extend({**r, "requested_norm": r["requested_residual_norm"], "rounded_norm": r["actual_rounded_residual_norm"]}
                            for r in self.component.records[before:])
        return result

    def __enter__(self):
        self.component.__enter__(); return self

    def __exit__(self, *args):
        self.component.__exit__(*args)


def make_patch(wm, candidate, donors, bank, sham=False, identity=False):
    if candidate["family"] == "coordinate":
        return SearchPatch(wm, candidate, donors, sham, identity)
    if bank is None:
        raise RuntimeError("Broad candidate needs frozen discovery bank")
    return BroadPatch(wm, candidate, bank, sham, identity)


class CombinedPatch:
    """Two component budgets sum to one common L1-in-time residual budget."""
    def __init__(self, wm, components, bank, selected=(0, 1), full_single=False, sham=False, targets=None):
        if len(components) != 2 or any(c["family"] == "coordinate" for c in components):
            raise ValueError("Interaction requires two broad native components")
        self.patches = []
        self.native = wm.unroll
        self.total_budget = min(float(bank[str(c["block"])+"/cap_radius"])*c["dose"] for c in components)
        self.records = []
        for index in selected:
            c = components[index]
            patch = BroadPatch(wm, c, bank, sham=sham)
            count = 6 if c["pulse"] == 0 else 1
            share = 1. if full_single else .5
            patch.component.cap_radius = self.total_budget*share/(c["dose"]*count)
            patch.component.norm_targets = None if targets is None else targets[index]
            self.patches.append((index, patch))

    def unroll(self, *args, **kwargs):
        actions = native_unroll_actions(args, kwargs)
        for index, patch in self.patches:
            if patch.candidate["family"] == "action_gain":
                patch.raw = patch.wm.preprocessor.denormalize_actions(actions.reshape(-1, 4).cpu()).to(actions).reshape(*actions.shape[:2], 5, 4)
            patch.component.reset()
        result = self.native(*args, **kwargs)
        self.records = [{"component": index, **row} for index, patch in self.patches for row in patch.component.records]
        total = sum(float(row["requested_residual_norm"].sum()) for row in self.records)
        if total > self.total_budget*actions.shape[1]+1e-3:
            raise RuntimeError("Interaction exceeded fixed total dose across all sites/times")
        return result

    def __enter__(self):
        self.stack = contextlib.ExitStack()
        for _, patch in self.patches:
            self.stack.enter_context(patch.component)
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


def interaction_delta(baseline, first, second, combined):
    return combined.double()-first.double()-second.double()+baseline.double()


@torch.no_grad()
def interactions(args):
    sys.path.insert(0, str(args.repo))
    from model_loader import load_headless_metaworld
    from evals.simu_env_planning.planning.utils import make_td
    from capture_specificity_controls import parameters_sha
    if not args.pairs or not args.bases or any(not 0 <= ep < 8 for ep in args.episodes):
        raise RuntimeError("Interactions need fixed pairs, discovery bases and development episodes0..7")
    pairs = json.loads(args.pairs.read_text())["pairs"]
    if len(pairs) > 4:
        raise RuntimeError("Bounded interaction shortlist at most4 pairs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"phase": "adaptive_development_component_interactions", "pairs": pairs, "episodes": args.episodes,
                "pairs_sha256": file_sha256(args.pairs), "basis_sha256": file_sha256(args.bases), "script_sha256": file_sha256(Path(__file__)),
                "total_dose": "Shared C=min(parent per-pulse maximum); sum residualL2 budgets over both components AND all active imagined times <=C. A/B-half each getC/2, ABsameC, A/B-full getC alone.",
                "interaction": "AB-A_half-B_half+unsteered in actual returnedvisual tensors; this is finite-dose network interaction, not physical utility", "future_truth_online": False}
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    wm, _, model = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False); before = parameters_sha(wm)
    bank = {k: torch.as_tensor(v, device="cuda", dtype=torch.float32) for k, v in np.load(args.bases, allow_pickle=False).items()}
    outputs, rows, started = [], [], time.monotonic()
    for episode in args.episodes:
        ref, source_sha = load_reference(args.reference_dirs, episode)
        z = wm.encode(make_td(ref["raw_context_visual"].clone(), {"proprio": ref["raw_context_proprio"].clone()}).cuda().unsqueeze(0), act=True)
        actions = ref["normalized_actions"].cuda()[:, None]
        baseline = visual(wm.unroll(z.clone(), act_suffix=actions))
        exact(visual(wm.unroll(z.clone(), act_suffix=actions)), baseline, "interaction unchanged repeated baseline")
        actual = torch.cat(ref["truth"]["encoded_visual"], 0).cuda().reshape_as(baseline)
        for pair in pairs:
            values, records = {"baseline": baseline}, {}
            targets = None
            for arm, selected, full, sham in (("A_half",(0,),False,False),("B_half",(1,),False,False),
                    ("AB",(0,1),False,False),("AB_sham",(0,1),False,True),("A_full",(0,),True,False),("B_full",(1,),True,False)):
                with CombinedPatch(wm, pair["components"], bank, selected, full, sham, targets if sham else None) as patch:
                    values[arm] = visual(patch.unroll(z.clone(), act_suffix=actions))
                records[arm] = patch.records
                if arm == "AB":
                    targets = {index: {r["step"]: r["requested_residual_norm"] for r in patch.records if r["component"] == index} for index in (0, 1)}
            delta = interaction_delta(values["baseline"], values["A_half"], values["B_half"], values["AB"])
            mse = {arm: (value-actual).square().mean((1, 2)).cpu().tolist() for arm, value in values.items()}
            norm = delta.flatten(1).norm(dim=-1)
            denominator = ((values["A_half"]-baseline).flatten(1).norm(dim=-1)+(values["B_half"]-baseline).flatten(1).norm(dim=-1)).double()
            row = {"pair_id": pair["id"], "episode": episode, "components": pair["components"], "source_sha256": source_sha,
                   "mse": mse, "interaction_raw_l2": norm.cpu().tolist(), "interaction_denominator_l2": denominator.cpu().tolist(),
                   "interaction_ratio": (norm/denominator.clamp_min(1e-9)).cpu().tolist(),
                   "total_budget": patch.total_budget, "actual_requested_total_norms": {k: sum(float(r["requested_residual_norm"].sum()) for r in v) for k, v in records.items()}}
            file = args.output_dir/f"episode-{episode:03d}-{pair['id']}.pt"
            torch.save({"row": row, "interaction_delta": delta.float().cpu(), "records": records}, file)
            outputs.append({"path": file.name, "sha256": file_sha256(file), "bytes": file.stat().st_size}); rows.append(row)
            print(json.dumps({"event": "search_interaction_complete", "episode": episode, "pair_id": pair["id"]}), flush=True)
    if parameters_sha(wm) != before:
        raise RuntimeError("Frozen interaction weights changed")
    report = args.output_dir/"interactions.json"
    write_json_atomic(report, {"complete": True, "rows": rows, "protocol": protocol, "seconds": time.monotonic()-started})
    outputs.extend({"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json"))
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})


class SearchPatch:
    def __init__(self, wm, candidate, donors, sham=False, identity=False):
        self.wm, self.candidate, self.donors, self.sham, self.identity = wm, candidate, donors, sham, identity
        self.block = wm.model.predictor.predictor_blocks[candidate["block"]]
        self.native = wm.unroll; self.records = []

    def hook(self, module, inputs, output):
        self.step += 1
        c = self.candidate
        if self.identity or c["pulse"] not in (0, self.step):
            return output
        if output.ndim != 3 or output.shape[1]%256 or output.shape[-1] != 400:
            raise RuntimeError("Unexpected native residual axes")
        current = output[:, -256:].float()
        if c["family"] != "coordinate" or c["site"] != "residual" or c["block"] != 3:
            raise RuntimeError("Pilot runtime accepts only verified P3 coordinate controls")
        pooled, metadata = self.donors.delta(current.mean(1), c["sign"], False)
        delta = pooled[:, None, :].expand(-1, 256, -1)*c["dose"]
        delta = masked_equal_energy(delta, token_mask(c["mask"], delta.device))
        if self.sham:
            delta = signed_permutation_sham(delta, 2026090603)
        edited = output.clone(); edited[:, -256:] += delta.to(output)
        rounded = edited[:, -256:].float()-current
        self.records.append({"step": self.step, "requested_norm": delta.flatten(1).norm(dim=-1).cpu(),
                             "rounded_norm": rounded.flatten(1).norm(dim=-1).cpu(),
                             "donor_available": metadata["donor_available"].cpu()})
        return edited

    def unroll(self, *args, **kwargs):
        self.step = 0
        result = self.native(*args, **kwargs)
        if self.step != 6:
            raise RuntimeError("Expected six native imagined steps")
        return result

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *_):
        self.handle.remove()


def visual(prediction):
    return prediction["visual"][1:, 0].float().reshape(6, 256, 384)


def screen_actions(actions, preprocessor, project_xyz=False):
    if not project_xyz:
        return actions, {"enabled": False}
    from run_xyz_projected_planner_v1 import project_actions
    projected, stats = project_actions(actions, preprocessor)
    return projected, {"enabled": True, **stats}


def physical_arm_specs(candidates, projected_five_arm=False):
    if projected_five_arm:
        if len(candidates) != 1:
            raise ValueError("Projected factorial comparison requires exactly one frozen candidate")
        c = candidates[0]
        return [(False, None, False), (True, None, False), (True, c, False), (True, c, True), (False, c, False)]
    return [(False, None, False)]+[(False, c, sham) for c in candidates for sham in (False, True)]


def physical_steps(env, actions):
    from collect_on_policy_bank import physics_snapshot
    from precompute_native_replay import ReachControlCapture
    base = env.proprio_env.unwrapped
    states, frames, controls, physics, rewards, native_info = [], [], [], [], [], []
    with ReachControlCapture(base) as capture:
        for action in actions:
            capture.records.clear()
            images, reward, done, infos = env.step_multiple(action[None].cpu())
            if len(images) != 1:
                raise RuntimeError("Native short fork ended before15 actions")
            states.append(np.asarray(infos[0]["state"]).copy()); frames.append(images[0].cpu())
            controls.append({"xyz_setter": list(capture.records), "actuator_ctrl": np.asarray(base.data.ctrl).copy(),
                             "mocap_position": np.asarray(base.data.mocap_pos).copy()})
            physics.append(physics_snapshot(env)); rewards.append(float(reward[0]))
            # Read the current native reward/success definition without stepping
            # physics again; unlike older captures this retains real info.success.
            _, info = base.evaluate_state(base._get_obs(), action.detach().cpu().numpy())
            native_info.append({k: float(v) for k, v in info.items() if np.asarray(v).shape == ()})
    return {"states": np.stack(states), "frames": torch.stack(frames), "controls": controls, "physics": physics,
            "raw_commands": actions.cpu(), "rewards": rewards, "native_info": native_info}


@torch.no_grad()
def physical(args):
    sys.path.insert(0, str(args.repo))
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from capture_horizon_coordinates import reconstruct_fresh, same_physics
    from causal_planner_forks import setup_cfg, close_env, visual_pooled
    from package_steering_banks import load_development
    from precompute_native_replay import safe_rows
    from collect_on_policy_bank import physics_snapshot
    from run_residual_coordinate_time import hand_goal_metrics
    from run_xyz_projected_planner_v1 import unroll_with_projection, project_actions
    if not args.candidates or not args.project_root or not args.inventory or not args.config:
        raise RuntimeError("Physical stage needs explicit frozen candidates/config/development inventory")
    candidates = json.loads(args.candidates.read_text())["candidates"]
    if len(candidates) > 4:
        raise RuntimeError("Bounded physical shortlist at most4 candidates")
    if any(not 0 <= episode < (12 if args.allow_seen_validation else 8) for episode in args.episodes):
        raise RuntimeError("Held physical episode refused before bank access")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = safe_rows(json.loads(args.inventory.read_text()), "reach_wall", args.episodes)
    protocol = {"phase": "adaptive_development_nativeCEM15step", "episodes": args.episodes, "candidates": candidates,
                "candidates_sha256": file_sha256(args.candidates), "broad_basis_sha256": file_sha256(args.bases) if args.bases else None,
                "script_sha256": file_sha256(Path(__file__)), "config_sha256": file_sha256(args.config),
                "dose_policy": "broad base mappedcap then dose; matchedsham uses sameinput local semanticnorm duringadaptiveCEM, first300 actions exact",
                "selection": "Adaptive development only; no untouched confirmation", "physical_steps": 15}
    protocol["projected_five_arm"] = getattr(args, "projected_physical_five_arm", False)
    protocol["projection_semantics"] = "If enabled, project ONLYmodelinputs rawXYZ[-1,1] with official affine statistics; gripper/nominalCEMproposals/execution unchanged. Native and projected baselines separate."
    protocol["projection_code_sha256"] = file_sha256(Path(__file__).with_name("run_xyz_projected_planner_v1.py"))
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    wm, preprocessor, model = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False); before = parameters_sha(wm)
    if any(r["model"]["checkpoint_sha256"] != file_sha256(Path(model["checkpoint"])) for r in rows):
        raise RuntimeError("Actual frozen checkpoint differs from development banks")
    cfg = setup_cfg(args.config, args.output_dir, wm)
    artifact = dict(np.load(args.coordinate, allow_pickle=False)); readout = dict(np.load(args.readout, allow_pickle=False))
    donors = DiscoveryDonors(artifact, 8, "cuda")
    bank = None if not args.bases else {k: torch.as_tensor(v, device="cuda", dtype=torch.float32) for k, v in np.load(args.bases, allow_pickle=False).items()}
    outputs, results, started = [], [], time.monotonic()
    for episode in args.episodes:
        source = next(r for r in rows if r["episode"] == episode)
        source_bank = load_development(args.project_root, source)
        start_hand = source_bank["replans"][0]["observation_proprio"].reshape(-1)[:3].numpy()
        baseline, projected_baseline = None, None
        arms = physical_arm_specs(candidates, getattr(args, "projected_physical_five_arm", False))
        for projected, candidate, sham in arms:
            if time.monotonic()-started > args.seconds_limit:
                raise RuntimeError("Per-worker physical budget reached before next independent arm")
            arm = "unsteered" if candidate is None else candidate["id"]+("-sham" if sham else "-edit")
            if projected:
                arm = "projected-"+arm
            env, agent, z, checks = reconstruct_fresh(cfg, wm, preprocessor, source_bank)
            initial = physics_snapshot(env); native = agent.planner.unroll; first = {}
            patch = None if candidate is None else make_patch(wm, candidate, donors, bank, sham=sham)
            if patch is not None:
                patch.native = native
            if baseline is not None:
                same_physics(initial, baseline["initial_physics"], "search paired physical initial")
                if checks["fresh_goal_sha256"] != baseline["fresh_goal_sha256"]:
                    raise RuntimeError("Paired goal changed")
                for key in ("visual", "proprio"):
                    exact(z[key], baseline["encoded_context"][key], "physical paired context "+key)
            def tracked(*a, **kw):
                actions = native_unroll_actions(a, kw)
                prediction = unroll_with_projection(native if patch is None else patch.unroll, preprocessor, projected, *a, **kw)
                if actions.shape[1] == 300 and not first:
                    costs = agent.objective(prediction, actions)
                    if projected and not torch.equal(costs, agent.objective(prediction, -actions)):
                        raise RuntimeError("Native objective unexpectedly depends on proposal action argument")
                    first.update(actions=actions.cpu().clone(), costs=costs.cpu(),
                                 pooled_visual=visual_pooled(prediction, actions).cpu())
                    if projected:
                        first["input_projection"] = project_actions(actions, preprocessor)[1]
                return prediction
            agent.planner.unroll = tracked
            try:
                print(json.dumps({"event": "search_physical_arm_started", "episode": episode, "arm": arm}), flush=True)
                with patch if patch is not None else contextlib.nullcontext():
                    prefix = agent.plan(z.clone(), steps_left=source_bank["replans"][0]["steps_left_model"])
                if baseline is not None:
                    exact(first["actions"], baseline["first_candidates"]["actions"], "search same first300 candidate actions")
                raw = preprocessor.denormalize_actions(prefix.cpu().reshape(15, 4))
                value = {"episode": episode, "arm": arm, "candidate": candidate, "sham": sham,
                         "model_input_xyz_projected": projected,
                         "initial_physics": initial, "encoded_context": {k: z[k].cpu() for k in ("visual", "proprio")},
                         "fresh_goal_sha256": checks["fresh_goal_sha256"], "first_candidates": first,
                         "selected_full_plan": agent.planner._prev_mean.detach().cpu(), "selected_raw_prefix": raw,
                         "selected_prediction": agent._predicted_best_encs_over_iterations[-1].detach().cpu(),
                         "fork": physical_steps(env, raw), "edit_records": [] if patch is None else patch.records}
                if baseline is None:
                    baseline = value
                if projected and candidate is None:
                    projected_baseline = value
                metrics = hand_goal_metrics(start_hand, value["fork"]["states"])
                base_metrics = hand_goal_metrics(start_hand, baseline["fork"]["states"])
                selected = value["selected_prediction"]["visual"][1:].float().reshape(6, -1, 384).mean(1)
                native_infos = value["fork"]["native_info"]
                result = {"episode": episode, "arm": arm, "candidate": candidate, "sham": sham, **metrics,
                          "model_input_xyz_projected": projected,
                          "progress_vs_baseline_m": metrics["hand_goal_progress_m"]-base_metrics["hand_goal_progress_m"],
                          "progress_vs_projected_baseline_m": None if projected_baseline is None else metrics["hand_goal_progress_m"]-hand_goal_metrics(start_hand, projected_baseline["fork"]["states"])["hand_goal_progress_m"],
                          "actual_hand_xyz_shift_m": (value["fork"]["states"][-1, :3]-baseline["fork"]["states"][-1, :3]).tolist(),
                          "raw_action_xyz_sum_change": (raw[:, :3].sum(0)-baseline["selected_raw_prefix"][:, :3].sum(0)).tolist(),
                          "first300_argmin": int(first["costs"].argmin()),
                          "first300_rank_changes": int((first["costs"].argsort().argsort()!=baseline["first_candidates"]["costs"].argsort().argsort()).sum()),
                          "same_first300_actions_exact": True, "own_selected_plan_decoded_xyz": decode_physical(selected, readout).tolist(),
                          "native_ever_success": any(r.get("success", 0) > 0 for r in native_infos),
                          "native_info_success_recorded": all("success" in r for r in native_infos),
                          "native_reward_sum": sum(value["fork"]["rewards"]), "native_reward_max": max(value["fork"]["rewards"]),
                          "seconds_elapsed": time.monotonic()-started}
                value["metrics"] = result
                path = args.output_dir/f"episode-{episode:03d}-{arm}.pt"; torch.save(value, path)
                receipt = {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
                write_json_atomic(path.with_suffix(".DONE.json"), {"complete": True, "outputs": [receipt]})
                outputs.append(receipt); results.append(result)
                print(json.dumps({"event": "search_physical_arm_complete", "metrics": result}), flush=True)
            finally:
                agent.planner.unroll = native; close_env(env)
    if parameters_sha(wm) != before:
        raise RuntimeError("Frozen physical-search weights changed")
    report = args.output_dir/"physical.json"
    write_json_atomic(report, {"complete": True, "rows": results, "protocol": protocol, "seconds": time.monotonic()-started,
                              "parameters_sha256": before, "model": model})
    outputs.extend({"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json"))
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})


@torch.no_grad()
def screen(args):
    sys.path.insert(0, str(args.repo))
    from model_loader import load_headless_metaworld
    from evals.simu_env_planning.planning.utils import make_td
    from capture_specificity_controls import parameters_sha
    candidates = json.loads(args.candidates.read_text())["candidates"] if args.candidates else (broad_candidates() if args.broad else pilot_candidates())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {"phase": "development_fixed_action_screen", "episodes": args.episodes, "candidates": candidates,
                "candidate_basis_sha256": file_sha256(args.coordinate), "script_sha256": file_sha256(Path(__file__)),
                "selection": "Actual encoded future error and matched sham specificity; probe shifts not selection",
                "future_truth_used_online": False, "sham": "fixed random signed channel permutation, same raw Frobenius norm and token support",
                "mask_policy": "coordinatepilot restores raw energy aftermask; broad masksbeforepostprojection/gatecap, doseAFTERbasecap; requested/rounded norms retained",
                "broad_basis_sha256": file_sha256(args.bases) if args.bases else None,
                "project_xyz_model_inputs": getattr(args, "project_xyz_actions", False),
                "input_projection_semantics": "If enabled, official affine action coordinates; clip rawXYZ only to[-1,1], gripper untouched; original simulator truth/commands preserved. Separate branch, no rewritten native screen.",
                "validation": "8..11 previously seen development-validation only after generation freeze", "untouched_held_access": False}
    if getattr(args, "project_xyz_actions", False):
        protocol["projection_code_sha256"] = file_sha256(Path(__file__).with_name("run_xyz_projected_planner_v1.py"))
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    wm, preprocessor, model = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False); before = parameters_sha(wm)
    artifact = dict(np.load(args.coordinate, allow_pickle=False)); readout = dict(np.load(args.readout, allow_pickle=False))
    donors = DiscoveryDonors(artifact, 8, "cuda")
    bank = None if not args.bases else {k: torch.as_tensor(v, device="cuda", dtype=torch.float32) for k, v in np.load(args.bases, allow_pickle=False).items()}
    rows, outputs, started = [], [], time.monotonic()
    for episode in args.episodes:
        if not 0 <= episode < (12 if args.allow_seen_validation else 8):
            raise RuntimeError("Episode excluded before tensor load")
        reference, source_sha = load_reference(args.reference_dirs, episode)
        z = wm.encode(make_td(reference["raw_context_visual"].clone(), {"proprio": reference["raw_context_proprio"].clone()}).cuda().unsqueeze(0), act=True)
        actions = reference["normalized_actions"].cuda()[:, None]
        original_input_baseline = None
        if getattr(args, "project_xyz_actions", False):
            original_input_baseline = wm.unroll(z.clone(), act_suffix=actions)
        actions, projection_stats = screen_actions(actions, preprocessor, getattr(args, "project_xyz_actions", False))
        original = {k: z[k].clone() for k in z.keys()}
        baseline = wm.unroll(z.clone(), act_suffix=actions)
        with make_patch(wm, candidates[0], donors, bank, identity=True) as patch:
            identity = patch.unroll(z.clone(), act_suffix=actions)
        for key in ("visual", "proprio"):
            exact(identity[key], baseline[key], "search full no-op identity "+key)
        actual = torch.cat(reference["truth"]["encoded_visual"], 0).cuda().reshape(6, 256, 384)
        truth = reference["truth"]["states"][5::5, :3]
        baseline_mse = (visual(baseline)-actual).square().mean((1, 2)).cpu().tolist()
        for candidate in candidates:
            row = {"candidate_id": candidate["id"], "episode": episode, "candidate": candidate, "baseline_mse": baseline_mse,
                   "source_sha256": source_sha, "identity_exact": True, "status": "complete", "complete": True}
            if original_input_baseline is not None:
                row["input_projection"] = projection_stats
                row["original_input_baseline_mse"] = (visual(original_input_baseline)-actual).square().mean((1, 2)).cpu().tolist()
            try:
                norms, targets = [], None
                for sham in (False, True):
                    with make_patch(wm, candidate, donors, bank, sham=sham) as patch:
                        if sham and isinstance(patch, BroadPatch):
                            patch.norm_targets = targets
                        prediction = patch.unroll(z.clone(), act_suffix=actions)
                    prefix = candidate["pulse"] or 1
                    for key in ("visual", "proprio"):
                        exact(prediction[key][:prefix], baseline[key][:prefix], "search pre-edit identity "+key)
                        exact(z[key], original[key], "search unchanged context "+key)
                    features = visual(prediction)
                    row["sham_mse" if sham else "edited_mse"] = (features-actual).square().mean((1, 2)).cpu().tolist()
                    row["sham_observer_error" if sham else "observer_error"] = np.linalg.norm(decode_physical(features.mean(1).cpu().numpy(), readout)-truth, axis=1).tolist()
                    norms.append(torch.cat([r["requested_norm"] for r in patch.records]))
                    if not sham:
                        targets = {r["step"]: r["requested_norm"] for r in patch.records}
                    row["actual_rank"] = getattr(patch, "actual_rank", 8)
                    row["sham_requested_norms" if sham else "edit_requested_norms"] = norms[-1].tolist()
                    row["sham_rounded_norms" if sham else "edit_rounded_norms"] = torch.cat([r["rounded_norm"] for r in patch.records]).tolist()
                row["normmatch_max_abs"] = float((norms[0]-norms[1]).abs().max())
                if not torch.allclose(norms[0], norms[1], atol=1e-5, rtol=1e-6):
                    raise RuntimeError("Fixed-pulse norm matched sham failed")
            except Exception as exc:
                row.update(status="failed", complete=False, failed=True, error=str(exc))
            rows.append(row)
            print(json.dumps({"event": "search_candidate_complete", "episode": episode, "candidate_id": candidate["id"], "status": row["status"]}), flush=True)
        path = args.output_dir/f"episode-{episode:03d}.json"
        write_json_atomic(path, {"complete": True, "rows": [r for r in rows if r["episode"] == episode]})
        outputs.append({"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    if parameters_sha(wm) != before:
        raise RuntimeError("Frozen weights changed")
    report = args.output_dir/"screen.json"
    write_json_atomic(report, {"complete": True, "rows": rows, "protocol": protocol, "model": model, "parameters_sha256": before,
                              "seconds": time.monotonic()-started, "candidate_failure_count": sum(r["status"] != "complete" for r in rows)})
    outputs.extend({"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (report, args.output_dir/"protocol.json"))
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "coordinate", "readout", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reference-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--allow-seen-validation", action="store_true")
    parser.add_argument("--bases", type=Path)
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--project-xyz-actions", action="store_true")
    parser.add_argument("--projected-physical-five-arm", action="store_true")
    parser.add_argument("--stage", choices=["screen", "physical", "interactions"], default="screen")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds-limit", type=float, default=3600)
    args = parser.parse_args()
    if args.project_xyz_actions and args.stage != "screen":
        parser.error("--project-xyz-actions is screen-only; physical projection requires explicit five-arm mode")
    if args.projected_physical_five_arm and args.stage != "physical":
        parser.error("--projected-physical-five-arm requires physical stage")
    torch.set_num_threads(2)
    try:
        {"physical": physical, "screen": screen, "interactions": interactions}[args.stage](args)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir/"FAILED.json").exists():
            write_json_atomic(args.output_dir/"FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
