#!/usr/bin/env python3
"""New off-shortlist-location and fixed-vector delayed-time diagnostics.

These controls are descriptive tests of a discovery candidate at one development
state. An off-shortlist layer is not assumed causally unrelated, and moving an
identical recorded edit tests intervention timing rather than stale-donor fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def match_raw_norm(delta, reference, epsilon=1e-12):
    """Preserve delta direction, match each candidate's reference raw L2 dose."""
    if delta.shape != reference.shape or delta.ndim != 2:
        raise ValueError("Norm matching needs equal [candidate,channel] shapes")
    if not torch.isfinite(delta).all() or not torch.isfinite(reference).all():
        raise ValueError("Non-finite raw edits")
    norm = delta.norm(dim=-1, keepdim=True)
    target = reference.norm(dim=-1, keepdim=True)
    if ((norm <= epsilon) & (target > epsilon)).any():
        raise ValueError("Cannot norm-match a zero off-site direction to a nonzero reference")
    ratio = torch.where(target <= epsilon, torch.zeros_like(norm), target / norm.clamp_min(epsilon))
    return delta * ratio


def fixed_pooled_delta(current, donor, basis, scale, beta=0.5):
    if beta == 0:
        return torch.zeros_like(current)
    return beta * ((((donor - current) / scale) @ basis) @ basis.T) * scale


class RecordedEditAtStep:
    """Apply one frozen raw vector to newest spatial tokens on one unroll call.

Steps are one-based future indices. At step two the predictor may contain the
old observed context plus newest input; only the final 256 output tokens change.
"""
    def __init__(self, block, unroll, delta, step):
        if step not in (1, 2) or delta.ndim != 2 or not torch.isfinite(delta).all():
            raise ValueError("Expected a finite [candidate,channel] edit at step1 or2")
        self.block, self.original_unroll, self.delta, self.target_step = block, unroll, delta.detach().clone(), step
        self.calls = 0
        self.edits = 0
        self.record = None
        self.handle = None

    def hook(self, _module, _inputs, output):
        self.calls += 1
        if self.calls != self.target_step:
            return output
        if (not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != len(self.delta)
                or output.shape[-1] != self.delta.shape[-1] or output.shape[1] % 256 != 0):
            raise ValueError("Expected aligned candidate batches and complete 256-token spatial frames")
        current = output[:, -256:].float().mean(1)
        if torch.count_nonzero(self.delta) == 0:
            edited = output  # bit-exact identity, even signed zeros and dtype.
        else:
            edited = output.clone()
            edited[:, -256:] = output[:, -256:] + self.delta.to(output.dtype).unsqueeze(1)
        if not torch.equal(edited[:, :-256], output[:, :-256]):
            raise RuntimeError("Delayed control altered old-context tokens")
        if not torch.isfinite(edited).all():
            raise RuntimeError("Non-finite edited activation")
        after = edited[:, -256:].float().mean(1)
        realized = after - current
        self.record = {"step": self.calls, "before_pooled": current.detach().cpu(),
                       "after_pooled": after.detach().cpu(), "requested_delta": self.delta.cpu(),
                       "realized_delta": realized.detach().cpu(),
                       "requested_raw_l2": self.delta.norm(dim=-1).cpu(),
                       "realized_raw_l2": realized.norm(dim=-1).cpu(),
                       "old_context_tokens_unchanged": True,
                       "token_count": output.shape[1]}
        self.edits += 1
        return edited

    def unroll(self, *args, **kwargs):
        self.calls, self.edits, self.record = 0, 0, None
        result = self.original_unroll(*args, **kwargs)
        if self.edits != 1:
            raise RuntimeError(f"Expected exactly one edit at step{self.target_step}; got {self.edits}")
        return result

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discovery_rows(receipt):
    if receipt.get("complete") is not True:
        raise ValueError("Incomplete discovery receipt")
    return [row for row in receipt["outputs"] if 0 <= int(row["episode"]) < 50]


def fit_off_shortlist_basis(capture_dirs):
    """Same five-probe/ridge/upper75%-motion fit as the frozen block3 candidate."""
    from fit_causal_subspace import fit_basis
    features, motion, seen, sources = [], [], set(), []
    for directory in capture_dirs:
        receipt_path = directory / "DONE.json"
        receipt = json.loads(receipt_path.read_text())
        sources.append({"path": str(receipt_path), "sha256": sha256(receipt_path)})
        for entry in discovery_rows(receipt):
            key = int(entry["seed"]), int(entry["episode"])
            if key in seen or key[0] not in (1, 2, 3):
                raise ValueError("Duplicate or invalid discovery episode")
            path = directory / entry["path"]
            if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != entry["sha256"]:
                raise ValueError("Unsafe path or discovery hash mismatch before tensor load")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            meta = payload["meta"]
            if meta["split"] != "discovery" or (int(meta["seed"]), int(meta["episode"])) != key:
                raise ValueError("Discovery payload metadata mismatch")
            features.append(payload["features"]["predictor"][:, 0].float().numpy())
            motion.append(payload["labels"]["realized_hand_delta"][:, [0, 2]].float().numpy())
            seen.add(key)
    if seen != {(seed, ep) for seed in (1, 2, 3) for ep in range(50)}:
        raise ValueError("Expected all150 discovery episodes")
    x, motion = np.concatenate(features).astype(float), np.concatenate(motion).astype(float)
    magnitude = np.linalg.norm(motion, axis=1)
    threshold = float(np.quantile(magnitude, .25))
    y = motion / np.maximum(magnitude[:, None], 1e-8)
    mean, scale, basis = fit_basis(x, y, magnitude > threshold, alpha=100., iterations=5)
    if basis.shape != (400, 10):
        raise ValueError(f"Predeclared rank10 off-site fit produced {basis.shape}; no adaptive refit")
    return {"block": 0, "mean": torch.tensor(mean, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32), "basis": torch.tensor(basis, dtype=torch.float32),
            "target": "realized_xz_direction", "sources": sources, "motion_mask_threshold": threshold,
            "ridge_alpha": 100., "probe_iterations": 5, "episodes": 150,
            "interpretation": "Off-shortlist control, not established causally unrelated"}


ARMS = ("unsteered", "identity", "block3_step1_up", "block3_step1_down",
        "block0_step1_up", "block0_step1_down", "block3_step2_up", "block3_step2_down")


def parameters_sha(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def decode_xyz(readout, features):
    return ((features - readout["mean"]) / readout["scale"]) @ readout["coef"].T + readout["intercept"]


@torch.no_grad()
def run(args):
    from causal_planner_forks import load_development_bank, setup_cfg, reconstruct, close_env, visual_pooled, emit
    from capture_causal_support import checked_artifact, require_exact, compare_float32_replay, ranks_by_step
    from collect_on_policy_bank import block_list
    from model_loader import load_headless_metaworld
    from protocol import write_json_atomic
    import time

    bank, bank_sha = load_development_bank(args.bank)
    receipt = json.loads((args.diagnostic_dir / "DONE.json").read_text())
    if (receipt.get("complete") is not True or bank["episode"] != 0 or receipt["episode"] != 0
            or receipt["replan"] != 0 or receipt["bank_sha256"] != bank_sha
            or receipt["block"] != 3 or receipt["rank"] != 10 or receipt["beta"] != .5):
        raise RuntimeError("Expected the frozen episode0/replan0/block3/rank10/beta0.5 diagnostic")
    candidate_sha = sha256(args.candidate)
    candidate_receipt = json.loads((args.candidate.parent / "DONE.json").read_text())
    if (candidate_receipt.get("complete") is not True or candidate_receipt["sha256"] != candidate_sha
            or candidate_sha != receipt["candidate_sha256"]
            or candidate_sha != "f4c295a148a22a9534cedd7d04b396f537bc8049842020886ba4f7fb0f1ca166"):
        raise RuntimeError("Original frozen block3 candidate SHA mismatch")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    baseline_cached = checked_artifact(args.diagnostic_dir, receipt, "unsteered.pt")
    counterfactuals = checked_artifact(args.diagnostic_dir, receipt, "counterfactuals.pt")
    actions_cpu = baseline_cached["first_candidates"]["actions"].clone()
    if tuple(actions_cpu.shape) != (6, 300, 20):
        raise RuntimeError("Expected immutable native H6/300/20 action candidates")
    decoder_receipt = json.loads((args.readout.parent / "DONE.json").read_text())
    decoder_entries = [row for row in decoder_receipt["outputs"] if row["path"] == args.readout.name]
    decoder_sha = sha256(args.readout)
    if decoder_receipt.get("complete") is not True or len(decoder_entries) != 1 or decoder_entries[0]["sha256"] != decoder_sha:
        raise RuntimeError("Frozen physical-readout checksum mismatch")
    with np.load(args.readout) as values:
        readout = {key: values[key].copy() for key in ("mean", "scale", "coef", "intercept")}
    control = fit_off_shortlist_basis(args.capture_dirs)
    if abs(control["motion_mask_threshold"] - candidate_receipt["motion_mask_threshold"]) > 1e-12:
        raise RuntimeError("Off-site discovery motion filter differs from the frozen candidate procedure")
    control_path = args.output_dir / "block0_control_basis.pt"
    torch.save(control, control_path)
    protocol = {
        "schema_version": 1, "scope": "One development state; same fixed candidates; no behavioral-specificity conclusion",
        "episode": 0, "replan": 0, "candidate_count": 300, "future_model_steps": 6,
        "arms": list(ARMS), "bank_sha256": bank_sha, "candidate_sha256": candidate_sha,
        "block3_rank": 10, "block0_rank": 10, "nominal_beta": .5,
        "offsite_basis_sha256": sha256(control_path), "offsite_fit": {key: control[key] for key in
            ("sources", "motion_mask_threshold", "ridge_alpha", "probe_iterations", "episodes")},
        "location_control": "Block0 is off-shortlist, not proven causally unrelated. Independent same-procedure basis; each requested raw delta norm matched to paired block3-step1 delta.",
        "time_control": "Identical recorded candidate-specific block3-step1 raw vector applied once to newest256 tokens of block3 unroll call2. No donor/current-h recomputation at step2.",
        "norm_policy": "Requested raw norms match per candidate; measured rounded pooled activation norms reported separately; zero off-site direction with nonzero reference aborts.",
        "time_index": "One-based imagined future1..6; earlier context tokens untouched",
        "goal": "Immutable cached raw visual/proprio goal; exact encoded goal at original saved dtype",
        "cross_process_replay": {"max_absolute_error": 1e-6, "applies_to": ["costs", "pooled_visual", "proprio"],
                                 "unsteered_ranks_and_argmin": "exact", "input_and_physics_checks": "existing replay guards unchanged",
                                 "native_cached_plan_bound": 1e-5, "native_plan_test": "not rerun; no CEM in this capture",
                                 "interpretation": "Float32 process-replay tolerance, not a scientific effect threshold"},
        "within_run_identity": "Bit-exact complete native visual/proprio traces, costs and action inputs",
        "decoder_sha256": decoder_sha, "new_cem_optimizations": 0, "new_physical_forks": 0,
        "diagnostic_done_sha256": sha256(args.diagnostic_dir / "DONE.json"),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                           ("causal_planner_forks.py", "capture_causal_support.py", "subspace_control.py", "fit_causal_subspace.py", "model_loader.py")},
        "accepted_operator": None,
    }
    protocol_path = args.output_dir / "protocol.json"
    write_json_atomic(protocol_path, protocol)
    emit("specificity_protocol_frozen", protocol_sha256=sha256(protocol_path), basis_sha256=sha256(control_path), arms=len(ARMS))

    wm, preprocessor, provenance = load_headless_metaworld(args.repo)
    wm.eval()
    wm.requires_grad_(False)
    parameter_before = parameters_sha(wm)
    cfg = setup_cfg(args.config, args.output_dir, wm)
    _, blocks = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    env, agent, z, replay_checks = reconstruct(cfg, wm, preprocessor, bank, 0, cached_goal=True)
    if agent.objective.sum_all_diffs:
        raise RuntimeError("Expected the unchanged terminal-distance objective")
    actions = actions_cpu.cuda()
    basis3, scale3 = candidate["basis"].cuda(), candidate["scale"].cuda()
    basis0, scale0 = control["basis"].cuda(), control["scale"].cuda()
    original_z = {key: z[key].detach().clone() for key in z.keys()}

    # The original +/-Z action chunks identify donors at exactly the same state.
    donor_plans = counterfactuals["normalized_actions"].cuda()
    if donor_plans.shape != (1, 6, 20):
        raise RuntimeError("Unexpected predeclared donor candidate layout")
    donor0_capture = []
    def donor_hook(_module, _inputs, output):
        if output.shape != (6, 256, 400):
            raise RuntimeError("Unexpected block0 donor tensor")
        donor0_capture.append(output.float().mean(1).detach())
    handle = blocks[0].register_forward_hook(donor_hook)
    try:
        wm.unroll(z.clone(), act_suffix=donor_plans)
    finally:
        handle.remove()
    if len(donor0_capture) != 1:
        raise RuntimeError("Expected a single unedited block0 donor call")
    donors0 = {"up": donor0_capture[0][4:5], "down": donor0_capture[0][5:6]}
    donors3 = {name: counterfactuals["donors"][name].cuda() for name in ("up", "down")}

    # Save reference first-step activations; no candidate costs select the edits.
    first_activations = {}
    handles = []
    for block_id in (0, 3):
        def first_hook(_module, _inputs, output, block_id=block_id):
            if block_id not in first_activations:
                if output.shape != (300, 256, 400):
                    raise RuntimeError("Unexpected initial same-candidate block layout")
                first_activations[block_id] = output.float().mean(1).detach()
        handles.append(blocks[block_id].register_forward_hook(first_hook))
    try:
        baseline_prediction = wm.unroll(z.clone(), act_suffix=actions)
    finally:
        for handle in handles:
            handle.remove()
    edits3, edits0 = {}, {}
    for direction in ("up", "down"):
        edits3[direction] = fixed_pooled_delta(first_activations[3], donors3[direction], basis3, scale3)
        preliminary = fixed_pooled_delta(first_activations[0], donors0[direction], basis0, scale0)
        edits0[direction] = match_raw_norm(preliminary, edits3[direction])
    torch.save({"block3_step1_delta": {key: value.cpu() for key, value in edits3.items()},
                "block0_step1_delta": {key: value.cpu() for key, value in edits0.items()},
                "block3_donors": {key: value.cpu() for key, value in donors3.items()},
                "block0_donors": {key: value.cpu() for key, value in donors0.items()},
                "baseline_first_activations": {str(key): value.cpu() for key, value in first_activations.items()},
                "actions": actions_cpu}, args.output_dir / "recorded_edits.pt")
    baseline_pooled = visual_pooled(baseline_prediction, actions).cpu()
    baseline_costs = agent.objective(baseline_prediction, actions).cpu()
    baseline_proprio = baseline_prediction["proprio"].cpu()
    start_xyz = np.asarray(bank["replans"][0]["observation_proprio"]).reshape(-1)[:3]
    goal_xyz = np.asarray(baseline_cached["fork"]["states"])[0, -3:]
    baseline_xyz = decode_xyz(readout, baseline_pooled.numpy())
    baseline_goal_distance = np.linalg.norm(baseline_xyz - goal_xyz, axis=-1)
    rows = []
    identity_checks = {}
    for arm in ARMS:
        started = time.monotonic()
        block_id = 0 if arm.startswith("block0") else 3
        step = 2 if "step2" in arm else 1
        direction = "down" if arm.endswith("down") else "up"
        delta = torch.zeros_like(edits3["up"]) if arm in ("unsteered", "identity") else edits0[direction] if block_id == 0 else edits3[direction]
        emit("specificity_arm_started", arm=arm, block=block_id, imagined_step=step)
        if arm == "unsteered":
            prediction = baseline_prediction
            record = {"before_pooled": first_activations[3].cpu(), "after_pooled": first_activations[3].cpu(),
                      "requested_delta": delta.cpu(), "realized_delta": delta.cpu(),
                      "requested_raw_l2": delta.norm(dim=-1).cpu(), "realized_raw_l2": delta.norm(dim=-1).cpu(),
                      "step": 1, "old_context_tokens_unchanged": True, "token_count": 256}
        else:
            patch = RecordedEditAtStep(blocks[block_id], wm.unroll, delta, step)
            with patch:
                prediction = patch.unroll(z.clone(), act_suffix=actions)
            if patch.calls != 6 or patch.edits != 1:
                raise RuntimeError("Control must edit exactly once in six native future calls")
            record = patch.record
            if step == 1:
                require_exact(record["before_pooled"], first_activations[block_id].cpu(), f"{arm} initial block activation")
        require_exact(actions.cpu(), actions_cpu, f"{arm} unchanged candidate inputs")
        for key, expected_z in original_z.items():
            require_exact(z[key], expected_z, f"{arm} unchanged encoded start {key}")
        pooled = visual_pooled(prediction, actions).cpu()
        costs = agent.objective(prediction, actions).cpu()
        step_costs = agent.objective(prediction, actions, keepdims=True)[1:].cpu()
        if step_costs.shape != (6, 300):
            raise RuntimeError("Expected6x300 native costs")
        require_exact(step_costs[-1], costs, f"{arm} terminal-cost layout")
        cross_process = None
        cached_name = {"unsteered": "unsteered", "identity": "identity", "block3_step1_up": "subspace_up", "block3_step1_down": "subspace_down"}.get(arm)
        if cached_name:
            cached = baseline_cached if cached_name == "unsteered" else checked_artifact(args.diagnostic_dir, receipt, cached_name + ".pt")
            require_exact(cached["first_candidates"]["actions"], actions_cpu, f"{arm} historical fixed candidates")
            cross_process = {
                "costs": compare_float32_replay(costs, cached["first_candidates"]["costs"], f"{arm} costs"),
                "pooled_visual": compare_float32_replay(pooled, cached["first_candidates"]["predicted_visual_pooled"], f"{arm} pooled visual"),
                "proprio": compare_float32_replay(prediction["proprio"].cpu(), cached["first_candidates"]["predicted_proprio"], f"{arm} proprio")}
            if arm == "unsteered":
                require_exact(ranks_by_step(costs[None]), ranks_by_step(cached["first_candidates"]["costs"][None]), "unsteered historical ranks")
                require_exact(costs.argmin(), cached["first_candidates"]["costs"].argmin(), "unsteered historical argmin")
        if arm == "identity":
            for key in ("visual", "proprio"):
                require_exact(prediction[key], baseline_prediction[key], f"identity complete {key} trace")
                identity_checks[key + "_exact"] = True
            require_exact(costs, baseline_costs, "identity costs")
            identity_checks["costs_exact"] = True
        if step == 2:
            for key in ("visual", "proprio"):
                require_exact(prediction[key][1], baseline_prediction[key][1], f"{arm} unchanged first future {key}")
            require_exact(record["requested_delta"], edits3[direction].cpu(), f"{arm} unchanged recorded raw vector")
        if block_id == 0:
            norm_error = float((record["requested_raw_l2"] - edits3[direction].norm(dim=-1).cpu()).abs().max())
            torch.testing.assert_close(record["requested_raw_l2"], edits3[direction].norm(dim=-1).cpu(), rtol=1e-6, atol=1e-6)
        else:
            norm_error = 0.
        xyz = decode_xyz(readout, pooled.numpy())
        xyz_delta = xyz - baseline_xyz
        goal_progress_change = baseline_goal_distance - np.linalg.norm(xyz - goal_xyz, axis=-1)
        ranks = ranks_by_step(step_costs)
        baseline_ranks = ranks_by_step(agent.objective(baseline_prediction, actions, keepdims=True)[1:].cpu())
        proprio_future = prediction["proprio"][1:].float()
        proprio_delta = proprio_future - baseline_prediction["proprio"][1:].float()
        visual_delta_norm = (pooled - baseline_pooled).norm(dim=-1)
        output = {"arm": arm, "block": block_id, "imagined_edit_step": step, "actions": actions_cpu,
                  "pooled_visual": pooled, "terminal_costs": costs, "costs_by_step": step_costs, "ranks_by_step": ranks,
                  "activation": record, "decoded_hand_xyz": torch.from_numpy(xyz),
                  "decoded_hand_delta_vs_unsteered": torch.from_numpy(xyz_delta),
                  "decoded_goal_progress_change_vs_unsteered": torch.from_numpy(goal_progress_change),
                  "pooled_visual_delta_norm_by_step": visual_delta_norm,
                  "proprio_latent_delta_norm_by_step": proprio_delta.flatten(2).norm(dim=-1).cpu(),
                  "cross_process_replay": cross_process, "replay_checks": replay_checks,
                  "physical_interpretation": "Frozen-probe decoded coordinates, not native physical outputs, simulator outcomes or behavioral specificity"}
        torch.save(output, args.output_dir / (arm + ".pt"))
        row = {"arm": arm, "family": "predictor", "layer": block_id, "imagined_edit_step": step,
               "scope": "same-candidate one-development-state diagnostic", "seconds": time.monotonic() - started,
               "requested_raw_norm_mean": float(record["requested_raw_l2"].mean()),
               "realized_raw_norm_mean": float(record["realized_raw_l2"].mean()),
               "max_requested_norm_match_error": norm_error,
               "max_rounded_norm_error": float((record["requested_raw_l2"] - record["realized_raw_l2"]).abs().max()),
               "terminal_cost_mean_change": float((costs - baseline_costs).mean()),
               "terminal_mean_absolute_rank_change": float((ranks[-1] - baseline_ranks[-1]).abs().mean()),
               "initial_best_candidate": int(costs.argmin()), "unsteered_initial_best_candidate": int(baseline_costs.argmin()),
               "initial_best_candidate_changed": bool(costs.argmin() != baseline_costs.argmin()),
               "selected_actions": "Not measured; argmin of this initial300-candidate bank is not the final native CEM selection",
               "decoded_xyz_delta_mean_by_future_step": xyz_delta.mean(1).tolist(),
               "decoded_xyz_delta_mean_norm_by_future_step": np.linalg.norm(xyz_delta, axis=-1).mean(1).tolist(),
               "decoded_goal_progress_delta_mean_by_future_step": goal_progress_change.mean(1).tolist(),
               "mean_absolute_rank_change_by_future_step": (ranks - baseline_ranks).abs().mean(1).tolist(),
               "cross_process_replay": cross_process, "behavioral_specificity": "not_measured", "accepted_operator": None}
        rows.append(row)
        write_json_atomic(args.output_dir / "progress.json", {"complete": False, "rows": rows})
        emit("specificity_arm_complete", arm=arm, cost_change=row["terminal_cost_mean_change"], rank_change=row["terminal_mean_absolute_rank_change"])
        if arm != "unsteered":
            del prediction, output
    close_env(env)
    parameter_after = parameters_sha(wm)
    if parameter_after != parameter_before:
        raise RuntimeError("Frozen model weights changed")
    report = {**protocol, "complete": True, "rows": rows, "identity_checks": identity_checks,
              "model": provenance, "checkpoint_sha256": sha256(Path(provenance["checkpoint"])),
              "parameters_before_sha256": parameter_before, "parameters_after_sha256": parameter_after,
              "parameters_unchanged": True, "original_start_xyz": start_xyz.tolist(), "goal_xyz": goal_xyz.tolist(),
              "protocol_sha256": sha256(protocol_path), "control_basis_sha256": sha256(control_path),
              "numerical_interpretation": "Location/timing diagnostics and decoded proxy changes; no behavioral efficacy or causal-unrelatedness claim"}
    write_json_atomic(args.output_dir / "specificity_controls.json", report)
    outputs = sorted(args.output_dir.glob("*.pt")) + [protocol_path, args.output_dir / "specificity_controls.json"]
    write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "episode": 0, "replan": 0,
                      "arms": list(ARMS), "script_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(protocol_path),
                      "outputs": [{"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]})
    emit("specificity_complete", arms=len(rows), done_sha256=sha256(args.output_dir / "DONE.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "bank", "candidate", "diagnostic-dir", "readout", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture-dirs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        (args.output_dir / "FAILED.json").write_text(json.dumps({"complete": False, "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
