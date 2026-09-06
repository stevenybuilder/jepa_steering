#!/usr/bin/env python3
"""Signed finite responses through the actual frozen JEPA-WM forward network.

One development state, immutable candidates, one saved raw direction. This is
neither a nonlinear probe nor a test of physical utility or manifold membership.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DOSES = (0., .125, -.125, .25, -.25, .5, -.5)
INDICES = tuple(range(8))


def finite_response(zero, plus, minus, beta):
    """Per-candidate directional finite differences, computed in float64.

The even/odd ratio is ||f(+b)+f(-b)-2f(0)|| / ||f(+b)-f(-b)||.
It measures finite-dose departure from odd linear response, not a Hessian
estimate independent of dose, physical utility, or statistical significance.
"""
    if beta <= 0 or zero.shape != plus.shape or zero.shape != minus.shape or zero.ndim < 1:
        raise ValueError("Positive dose and equal candidate-first tensors required")
    if not all(torch.isfinite(value).all() for value in (zero, plus, minus)):
        raise ValueError("Non-finite captured response")
    zero, plus, minus = (value.double().reshape(len(value), -1) for value in (zero, plus, minus))
    odd = plus - minus
    even = plus + minus - 2 * zero
    norm0, norm_odd, norm_even = (value.norm(dim=-1) for value in (zero, odd, even))
    def ratio(numerator, denominator):
        return [float(n / d) if float(d) > 1e-15 else None for n, d in zip(numerator, denominator)]
    return {
        "beta": beta, "first_derivative_l2": (norm_odd / (2 * beta)).tolist(),
        "second_derivative_l2": (norm_even / (beta * beta)).tolist(),
        "central_first_difference_l2": (norm_odd / 2).tolist(),
        "central_second_difference_l2": norm_even.tolist(),
        "baseline_l2": norm0.tolist(), "even_to_odd_norm_ratio": ratio(norm_even, norm_odd),
        "second_difference_relative_to_baseline": ratio(norm_even, norm0),
        "zero_odd_response_mask": (norm_odd <= 1e-15).tolist(),
        "plus_change_l2": (plus - zero).norm(dim=-1).tolist(),
        "minus_change_l2": (minus - zero).norm(dim=-1).tolist(),
    }


def first_difference_variation(plus, minus, beta, reference_plus, reference_minus, reference_beta=.125):
    """Compare full central derivative vectors, including direction changes."""
    if min(beta, reference_beta) <= 0 or any(value.shape != plus.shape for value in (minus, reference_plus, reference_minus)):
        raise ValueError("Aligned candidate-first tensors and positive doses required")
    if not all(torch.isfinite(value).all() for value in (plus, minus, reference_plus, reference_minus)):
        raise ValueError("Non-finite finite difference")
    current = (plus.double() - minus.double()).reshape(len(plus), -1) / (2 * beta)
    reference = (reference_plus.double() - reference_minus.double()).reshape(len(plus), -1) / (2 * reference_beta)
    refnorm, currentnorm = reference.norm(dim=-1), current.norm(dim=-1)
    error = (current - reference).norm(dim=-1)
    return {"beta": beta, "reference_beta": reference_beta,
            "reference_first_derivative_l2": refnorm.tolist(), "first_derivative_l2": currentnorm.tolist(),
            "derivative_vector_change_l2": error.tolist(),
            "normalized_derivative_vector_change": [float(e / r) if float(r) > 1e-15 else None for e, r in zip(error, refnorm)],
            "normalized_derivative_norm_change": [float((v - r).abs() / r) if float(r) > 1e-15 else None for v, r in zip(currentnorm, refnorm)],
            "zero_reference_mask": (refnorm <= 1e-15).tolist()}


def summarize_first_differences(source_dir, output_dir):
    """CPU-only postprocessing of completed immutable raw captures; no model."""
    from capture_specificity_controls import sha256
    from protocol import write_json_atomic
    receipt = json.loads((source_dir / "DONE.json").read_text())
    if receipt.get("complete") is not True:
        raise RuntimeError("Only complete interface captures may be summarized")
    entries = {row["path"]: row for row in receipt["outputs"]}
    report_path = source_dir / "interface_response.json"
    if sha256(report_path) != entries[report_path.name]["sha256"]:
        raise RuntimeError("Source report checksum mismatch")
    source_report = json.loads(report_path.read_text())
    captures = {}
    for row in source_report["dose_rows"]:
        beta, path = float(row["beta"]), source_dir / row["file"]
        if beta == 0:
            continue
        if not path.resolve().is_relative_to(source_dir.resolve()) or sha256(path) != entries[path.name]["sha256"]:
            raise RuntimeError("Raw interface checksum mismatch before tensor load")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if float(payload["beta"]) != beta or tuple(payload["candidate_indices"]) != INDICES:
            raise RuntimeError("Source candidate/dose mismatch")
        captures[beta] = payload["stages"]
    if set(captures) != set(DOSES) - {0.}:
        raise RuntimeError("Expected all six signed nonzero doses")
    rows = []
    for stage in source_report["interfaces"]:
        for beta in (.125, .25, .5):
            rows.append({"stage": stage, "family": "predictor", "layer": 3,
                         **first_difference_variation(captures[beta][stage], captures[-beta][stage], beta,
                                                      captures[.125][stage], captures[-.125][stage])})
    result = {"complete": True, "source_report_sha256": sha256(report_path),
              "source_done_sha256": sha256(source_dir / "DONE.json"), "reference_beta": .125,
              "reported_candidate_indices": list(INDICES), "episode_count": 1, "rows": rows,
              "metric": "||J_b-J_0.125||/||J_0.125||, J_b=(f(+b)-f(-b))/(2b); uses FULL flattened stage tensors, not only their norm; None if reference derivative is zero",
              "interpretation": "Directional finite-dose derivative variation; catches odd response changes missed by symmetric second differences; not a global linearity test or intrinsic manifold/physical utility claim",
              "new_model_forwards": 0, "device": "cpu", "script_sha256": sha256(Path(__file__))}
    output = output_dir / "first_difference_variation.json"
    write_json_atomic(output, result)
    write_json_atomic(output_dir / "DONE.json", {"complete": True, "outputs": [
        {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size}]})
    print(json.dumps({"event": "first_difference_summary_complete", "rows": len(rows), "sha256": sha256(output)}), flush=True)


class CaptureFirstInterfaces:
    """Capture first-call tensors only; keep all native batches unchanged."""
    def __init__(self, block, norm, projection, candidate_count=300, indices=INDICES):
        self.modules = {"p3_edited_residual": block, "predictor_norm": norm, "predictor_proj": projection}
        self.candidate_count, self.indices = candidate_count, tuple(indices)
        self.values, self.handles, self.counts = {}, [], {}

    def save(self, name, value):
        self.counts[name] = self.counts.get(name, 0) + 1
        if name in self.values:
            return
        valid_layout = isinstance(value, torch.Tensor) and (
            (value.ndim == 3 and value.shape[1] == 256)
            or (name.startswith("predictor_proj_") and value.ndim == 4 and value.shape[1:3] == (1, 256)))
        if not valid_layout or value.shape[0] != self.candidate_count:
            raise RuntimeError(f"Unexpected first-step native interface layout: {name} {getattr(value, 'shape', None)}")
        self.values[name] = value[list(self.indices)].detach().float().cpu().clone()

    def __enter__(self):
        for name, module in self.modules.items():
            if name != "p3_edited_residual":
                self.handles.append(module.register_forward_pre_hook(
                    lambda _m, args, name=name: self.save(name + "_input", args[0])))
            self.handles.append(module.register_forward_hook(
                lambda _m, _args, value, name=name: self.save(name if name == "p3_edited_residual" else name + "_output", value)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def source_checked(directory, name):
    from capture_causal_support import checked_artifact
    receipt = json.loads((directory / "DONE.json").read_text())
    return checked_artifact(directory, receipt, name)


@torch.no_grad()
def run(args):
    from capture_specificity_controls import RecordedEditAtStep, sha256, parameters_sha
    from capture_causal_support import compare_float32_replay, require_exact, ranks_by_step
    from causal_planner_forks import load_development_bank, setup_cfg, reconstruct, close_env, visual_pooled, emit
    from collect_on_policy_bank import block_list
    from model_loader import load_headless_metaworld
    from protocol import write_json_atomic

    bank, bank_sha = load_development_bank(args.bank)
    original = source_checked(args.diagnostic_dir, "unsteered.pt")
    reference = source_checked(args.specificity_dir, "recorded_edits.pt")
    reference_report = json.loads((args.specificity_dir / "specificity_controls.json").read_text())
    reference_done = json.loads((args.specificity_dir / "DONE.json").read_text())
    report_entry = [row for row in reference_done["outputs"] if row["path"] == "specificity_controls.json"]
    if len(report_entry) != 1 or sha256(args.specificity_dir / "specificity_controls.json") != report_entry[0]["sha256"]:
        raise RuntimeError("Specificity source report hash mismatch")
    if (bank["episode"] != 0 or reference_report["episode"] != 0 or reference_report["replan"] != 0
            or reference_report["bank_sha256"] != bank_sha or reference_report["nominal_beta"] != .5
            or reference_report["candidate_sha256"] != "f4c295a148a22a9534cedd7d04b396f537bc8049842020886ba4f7fb0f1ca166"):
        raise RuntimeError("Frozen episode/candidate/direction provenance mismatch")
    actions_cpu = original["first_candidates"]["actions"].clone()
    require_exact(reference["actions"], actions_cpu, "recorded direction action bank")
    if tuple(actions_cpu.shape) != (6, 300, 20):
        raise RuntimeError("Expected same original native300 candidates/H6")
    # Exact binary scaling restores nominal beta1 from the frozen beta.5 vector.
    raw_direction = reference["block3_step1_delta"]["up"] * 2
    if raw_direction.shape != (300, 400) or not torch.isfinite(raw_direction).all():
        raise RuntimeError("Expected finite saved rank10 pooled raw direction")
    torch.save({"raw_direction_beta1": raw_direction, "actions": actions_cpu, "candidate_indices": INDICES}, args.output_dir / "frozen_inputs.pt")
    protocol = {
        "schema_version": 1, "episode": 0, "replan": 0, "family": "predictor", "layer": 3,
        "block_edit_step": 1, "doses": list(DOSES), "reported_candidate_indices": list(INDICES),
        "native_candidate_count_every_run": 300, "future_model_steps": 6,
        "direction": "One immutable up donor-derived rank10 raw vector per candidate, saved at beta0.5 in specificity run and doubled to nominal beta1; negative doses reverse THIS SAME direction, not the down donor.",
        "capture": "Full256 tokens for first8 predetermined candidates, imagined step1 only; native batch300 remains unchanged at all doses",
        "native_layout": "Block/norm [B,256,400]; AdaLN projection keeps [B,T=1,256,384] after splitting16 proprio channels. Store native dimensions without reshaping stage captures.",
        "interfaces": ["p3_edited_residual", "predictor_norm_input", "predictor_norm_output", "predictor_proj_input", "predictor_proj_output", "returned_visual", "returned_proprio", "native_first_step_goal_loss"],
        "batch_policy": "No candidate batch reduction or across-batch exactness claim",
        "finite_difference_metric": "Per-candidate Frobenius/L2 in each stage's native coordinates; central first difference=(f(+b)-f(-b))/2; central second difference=f(+b)+f(-b)-2f(0); derivative-scaled versions divide by b and b^2; even/odd norm ratio=norm(second)/norm(f(+b)-f(-b)). Stage ratios have different native metrics; no coordinate-invariant localization claim.",
        "identity": "A separate unhooked baseline and beta0 complete native visual/proprio/cost traces must be bit exact",
        "cross_process_policy": "Original cached costs/pooled/proprio maxABS<=1e-6 and exact unsteered ranks/argmin; cached raw/encoded goal and simulator/start guards unchanged",
        "raw_direction_source_sha256": sha256(args.specificity_dir / "recorded_edits.pt"),
        "specificity_done_sha256": sha256(args.specificity_dir / "DONE.json"),
        "diagnostic_done_sha256": sha256(args.diagnostic_dir / "DONE.json"),
        "bank_sha256": bank_sha, "candidate_sha256": reference_report["candidate_sha256"],
        "frozen_inputs_sha256": sha256(args.output_dir / "frozen_inputs.pt"),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                           ("capture_specificity_controls.py", "causal_planner_forks.py", "capture_causal_support.py", "model_loader.py")},
        "new_cem_optimizations": 0, "new_physical_forks": 0, "held_episodes_loaded": 0,
        "interpretation": "Actual forward finite-dose nonlinearity diagnostic, not a learned nonlinear readout, manifold proof, physical utility, behavioral efficacy or accepted operator",
    }
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    emit("interface_protocol_frozen", protocol_sha256=sha256(args.output_dir / "protocol.json"), doses=list(DOSES), indices=list(INDICES))
    wm, preprocessor, provenance = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False)
    parameter_before = parameters_sha(wm)
    cfg = setup_cfg(args.config, args.output_dir, wm)
    env, agent, z, replay = reconstruct(cfg, wm, preprocessor, bank, 0, cached_goal=True)
    predictor = wm.model.predictor
    _, blocks = block_list(predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    norm, projection = predictor.predictor_norm, predictor.predictor_proj
    actions, direction = actions_cpu.cuda(), raw_direction.cuda()
    original_z = {key: z[key].clone() for key in z.keys()}
    baseline = wm.unroll(z.clone(), act_suffix=actions)
    baseline_costs = agent.objective(baseline, actions).cpu()
    cross_process = {
        "costs": compare_float32_replay(baseline_costs, original["first_candidates"]["costs"], "baseline costs"),
        "pooled_visual": compare_float32_replay(visual_pooled(baseline, actions).cpu(), original["first_candidates"]["predicted_visual_pooled"], "baseline pooled visual"),
        "proprio": compare_float32_replay(baseline["proprio"].cpu(), original["first_candidates"]["predicted_proprio"], "baseline proprio"),
    }
    require_exact(ranks_by_step(baseline_costs[None]), ranks_by_step(original["first_candidates"]["costs"][None]), "baseline historical ranks")
    require_exact(baseline_costs.argmin(), original["first_candidates"]["costs"].argmin(), "baseline historical argmin")
    captures, dose_rows = {}, []
    initial_hooks = {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in
                     (("block", blocks[3]), ("norm", norm), ("projection", projection))}
    for beta in DOSES:
        emit("interface_dose_started", beta=beta, native_candidates=300)
        patch = RecordedEditAtStep(blocks[3], wm.unroll, direction * beta, 1)
        capture = CaptureFirstInterfaces(blocks[3], norm, projection)
        # Registration order matters: residual capture must see patched output.
        with patch, capture:
            prediction = patch.unroll(z.clone(), act_suffix=actions)
        if patch.calls != 6 or patch.edits != 1 or any(count != 6 for count in capture.counts.values()):
            raise RuntimeError("Unexpected native first-step hook/unroll call counts")
        require_exact(patch.record["before_pooled"], reference["baseline_first_activations"]["3"], "same first-step unedited residual")
        require_exact(patch.record["requested_delta"], raw_direction * beta, "same saved raw direction at requested dose")
        require_exact(actions.cpu(), actions_cpu, "immutable action candidates")
        for key in original_z:
            require_exact(z[key], original_z[key], "immutable encoded start " + key)
        costs = agent.objective(prediction, actions).cpu()
        if beta == 0:
            for key in ("visual", "proprio"):
                require_exact(prediction[key], baseline[key], "beta0 full native " + key)
            require_exact(costs, baseline_costs, "beta0 complete terminal costs")
        values = capture.values
        values["returned_visual"] = prediction["visual"][1, list(INDICES)].float().cpu().clone()
        values["returned_proprio"] = prediction["proprio"][1, list(INDICES)].float().cpu().clone()
        first_loss = agent.objective(prediction, actions, keepdims=True)[1].cpu()
        if first_loss.shape != (300,):
            raise RuntimeError("Native first-imagined-step goal loss layout changed")
        values["native_first_step_goal_loss"] = first_loss[list(INDICES)].clone()
        if set(values) != set(protocol["interfaces"]):
            raise RuntimeError("Missing required actual-network interface")
        name = "zero" if beta == 0 else ("plus" if beta > 0 else "minus") + str(abs(beta)).replace(".", "p")
        file = args.output_dir / (name + ".pt")
        torch.save({"beta": beta, "candidate_indices": INDICES, "stages": values,
                    "activation": {key: value[list(INDICES)] if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == 300 else value for key, value in patch.record.items()},
                    "first_step_native_goal_losses_all300": first_loss,
                    "all300_terminal_costs": costs, "hook_calls": capture.counts}, file)
        captures[beta] = values
        dose_rows.append({"beta": beta, "file": file.name, "sha256": sha256(file),
                          "requested_raw_l2_first8": patch.record["requested_raw_l2"][list(INDICES)].tolist(),
                          "realized_raw_l2_first8": patch.record["realized_raw_l2"][list(INDICES)].tolist(),
                          "first_step_native_goal_loss_first8": first_loss[list(INDICES)].tolist(),
                          "stage_shapes": {name: list(value.shape) for name, value in values.items()}})
        emit("interface_dose_complete", beta=beta, output_sha256=sha256(file))
        del prediction
    for name, module in (("block", blocks[3]), ("norm", norm), ("projection", projection)):
        if initial_hooks[name] != (len(module._forward_hooks), len(module._forward_pre_hooks)):
            raise RuntimeError("Hook cleanup failed: " + name)
    close_env(env)
    parameter_after = parameters_sha(wm)
    if parameter_before != parameter_after:
        raise RuntimeError("Frozen parameters changed")
    rows = []
    for stage in protocol["interfaces"]:
        for beta in (.125, .25, .5):
            rows.append({"family": "predictor", "layer": 3, "stage": stage, "target": "actual_forward_directional_curvature",
                         **finite_response(captures[0.][stage], captures[beta][stage], captures[-beta][stage], beta)})
    report = {**protocol, "complete": True, "rows": rows, "dose_rows": dose_rows,
              "identity_exact": True, "hooks_cleaned": True, "parameters_unchanged": True,
              "parameters_before_sha256": parameter_before, "parameters_after_sha256": parameter_after,
              "replay_checks": replay, "cross_process_replay": cross_process, "model": provenance,
              "checkpoint_sha256": sha256(Path(provenance["checkpoint"])),
              "protocol_sha256": sha256(args.output_dir / "protocol.json")}
    write_json_atomic(args.output_dir / "interface_response.json", report)
    outputs = sorted(args.output_dir.glob("*.pt")) + [args.output_dir / "protocol.json", args.output_dir / "interface_response.json"]
    write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "script_sha256": sha256(Path(__file__)),
                      "outputs": [{"path": file.name, "sha256": sha256(file), "bytes": file.stat().st_size} for file in outputs]})
    emit("interface_response_complete", doses=len(DOSES), rows=len(rows), done_sha256=sha256(args.output_dir / "DONE.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "bank", "diagnostic-dir", "specificity-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarize-first-differences", type=Path)
    args = parser.parse_args()
    if not args.summarize_first_differences and any(getattr(args, name) is None for name in ("repo", "config", "bank", "diagnostic_dir", "specificity_dir")):
        parser.error("Forward capture requires repo/config/bank/diagnostic-dir/specificity-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.summarize_first_differences:
            summarize_first_differences(args.summarize_first_differences, args.output_dir)
        else:
            run(args)
    except Exception as exc:
        (args.output_dir / "FAILED.json").write_text(json.dumps({"complete": False, "error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
