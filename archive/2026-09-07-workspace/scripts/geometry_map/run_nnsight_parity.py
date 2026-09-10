#!/usr/bin/env python3
"""Parity only against immutable native-hook captures; no new causal evidence."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import torch

from nnsight_trace import NNsightPredictorTrace, MODULE_PATHS
from capture_specificity_controls import sha256, parameters_sha
from capture_causal_support import checked_artifact, require_exact, compare_float32_replay, ranks_by_step
from causal_planner_forks import load_development_bank, setup_cfg, reconstruct, close_env, visual_pooled, emit
from model_loader import load_headless_metaworld
from protocol import write_json_atomic

DOSES = (0., .125, -.125, .25, -.25, .5, -.5)
INDICES = tuple(range(8))


def compare_stage(actual, expected, label):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"Tracer parity shape/finiteness mismatch: {label}")
    error = float((actual.double() - expected.double()).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"Tracer parity {label}: {error} > fixed1e-6")
    return {"max_absolute_error": error, "exact": torch.equal(actual, expected),
            "bound": 1e-6, "actual_shape": list(actual.shape), "actual_dtype": str(actual.dtype)}


@torch.no_grad()
def run(args):
    bank, bank_sha = load_development_bank(args.bank)
    diagnostic = json.loads((args.diagnostic_dir / "DONE.json").read_text())
    reference_done = json.loads((args.reference_dir / "DONE.json").read_text())
    report = json.loads((args.reference_dir / "interface_response.json").read_text())
    report_entry = next(row for row in reference_done["outputs"] if row["path"] == "interface_response.json")
    if sha256(args.reference_dir / "interface_response.json") != report_entry["sha256"]:
        raise RuntimeError("Reference report SHA mismatch")
    if (not report["complete"] or bank["episode"] != 0 or report["episode"] != 0
            or report["bank_sha256"] != bank_sha or tuple(report["doses"]) != DOSES
            or report["candidate_sha256"] != "f4c295a148a22a9534cedd7d04b396f537bc8049842020886ba4f7fb0f1ca166"):
        raise RuntimeError("Expected unchanged development state/doses/candidate")
    frozen = checked_artifact(args.reference_dir, reference_done, "frozen_inputs.pt")
    original = checked_artifact(args.diagnostic_dir, diagnostic, "unsteered.pt")
    actions_cpu = frozen["actions"]
    require_exact(actions_cpu, original["first_candidates"]["actions"], "frozen native candidate actions")
    if tuple(actions_cpu.shape) != (6, 300, 20) or tuple(frozen["raw_direction_beta1"].shape) != (300, 400):
        raise RuntimeError("Unexpected frozen native layout")
    protocol = {"scope": "NNsight tracer parity against completed native-hook seven-dose diagnostic; NOT new independent causal evidence",
                "episode": 0, "replan": 0, "candidate_count": 300, "capture_indices": list(INDICES),
                "doses": list(DOSES), "imagined_edit_step": 1, "module_paths": MODULE_PATHS,
                "direction": "Exact saved beta1 up raw vector; signed doses reverse the same vector, not alternate donors",
                "reference_done_sha256": sha256(args.reference_dir / "DONE.json"),
                "reference_report_sha256": sha256(args.reference_dir / "interface_response.json"),
                "bank_sha256": bank_sha, "script_sha256": sha256(__file__),
                "tracer_script_sha256": sha256(Path(__file__).with_name("nnsight_trace.py")),
                "nnsight_version": importlib.metadata.version("nnsight"), "torch_version": torch.__version__,
                "parity_policy": "MaxABS<=1e-6 every saved stage/loss; unsteered ranks/argmin exact; within-run complete beta0 identity exact. No bound expansion on failure.",
                "official_docs": ["https://nnsight.net/getting-started/quickstart/", "https://nnsight.net/features/1_getting/", "https://pypi.org/project/nnsight/0.7.0/"],
                "new_cem_optimizations": 0, "new_simulator_action_forks": 0, "held_episodes_loaded": 0}
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    emit("nnsight_protocol_frozen", nnsight=protocol["nnsight_version"], torch=protocol["torch_version"])
    wm, preprocessor, provenance = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False)
    if sha256(Path(provenance["checkpoint"])) != report["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from native-hook reference")
    parameter_before = parameters_sha(wm)
    cfg = setup_cfg(args.config, args.output_dir, wm)
    env, agent, z, replay = reconstruct(cfg, wm, preprocessor, bank, 0, cached_goal=True)
    actions, direction = actions_cpu.cuda(), frozen["raw_direction_beta1"].cuda()
    start = {key: z[key].clone() for key in z.keys()}
    baseline = wm.unroll(z.clone(), act_suffix=actions)
    baseline_costs = agent.objective(baseline, actions).cpu()
    original_checks = {
        "costs": compare_float32_replay(baseline_costs, original["first_candidates"]["costs"], "native baseline costs"),
        "pooled": compare_float32_replay(visual_pooled(baseline, actions).cpu(), original["first_candidates"]["predicted_visual_pooled"], "native baseline pooled"),
        "proprio": compare_float32_replay(baseline["proprio"].cpu(), original["first_candidates"]["predicted_proprio"], "native baseline proprio")}
    require_exact(ranks_by_step(baseline_costs[None]), ranks_by_step(original["first_candidates"]["costs"][None]), "native baseline ranks")
    require_exact(baseline_costs.argmin(), original["first_candidates"]["costs"].argmin(), "native baseline argmin")
    hook_counts = {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in wm.named_modules()}
    tracer = NNsightPredictorTrace(wm, INDICES)
    rows = []
    for beta in DOSES:
        emit("nnsight_dose_started", beta=beta)
        traced = tracer.trace_unroll(z.clone(), actions, direction * beta)
        prediction = traced["prediction"]
        if beta == 0:
            for key in ("visual", "proprio"):
                require_exact(prediction[key], baseline[key], "NNsight beta0 complete " + key)
            require_exact(agent.objective(prediction, actions).cpu(), baseline_costs, "NNsight beta0 costs")
        reference_row = next(row for row in report["dose_rows"] if row["beta"] == beta)
        reference = checked_artifact(args.reference_dir, reference_done, reference_row["file"])
        stages = traced["stages"]
        stages["returned_visual"] = prediction["visual"][1, list(INDICES)].float().cpu().clone()
        stages["returned_proprio"] = prediction["proprio"][1, list(INDICES)].float().cpu().clone()
        first_loss = agent.objective(prediction, actions, keepdims=True)[1].cpu()
        costs = agent.objective(prediction, actions).cpu()
        stages["native_first_step_goal_loss"] = first_loss[list(INDICES)]
        comparisons = {stage: compare_stage(value, reference["stages"][stage], stage) for stage, value in stages.items()}
        comparisons["all300_first_step_losses"] = compare_stage(first_loss, reference["first_step_native_goal_losses_all300"], "all300 firststep losses")
        comparisons["all300_terminal_costs"] = compare_stage(costs, reference["all300_terminal_costs"], "all300 terminal costs")
        require_exact(actions.cpu(), actions_cpu, "unchanged candidate actions")
        for key in start:
            require_exact(z[key], start[key], "unchanged encoded start " + key)
        file = args.output_dir / reference_row["file"]
        torch.save({"beta": beta, "stages": stages, "module_paths": MODULE_PATHS,
                    "candidate_indices": INDICES, "before_residual": traced["before_residual"],
                    "all300_first_step_losses": first_loss, "all300_terminal_costs": costs,
                    "comparisons": comparisons}, file)
        rows.append({"beta": beta, "file": file.name, "sha256": sha256(file), "comparisons": comparisons})
        emit("nnsight_dose_complete", beta=beta, maximum_error=max(value["max_absolute_error"] for value in comparisons.values()))
        del traced, prediction, stages, reference
    tracer.close()
    parameter_after = parameters_sha(wm)
    if parameter_before != parameter_after:
        raise RuntimeError("Frozen model parameters changed")
    for name, module in wm.named_modules():
        if hook_counts[name] != (len(module._forward_hooks), len(module._forward_pre_hooks)):
            raise RuntimeError("NNsight hook cleanup failed at " + name)
    close_env(env)
    write_json_atomic(args.output_dir / "nnsight_parity.json", {**protocol, "complete": True, "rows": rows,
                      "within_run_identity_exact": True, "parameters_unchanged": True, "hooks_cleaned": True,
                      "parameters_before_sha256": parameter_before, "parameters_after_sha256": parameter_after,
                      "checkpoint_sha256": report["checkpoint_sha256"], "model": provenance,
                      "replay_checks": replay, "original_baseline_checks": original_checks})
    outputs = sorted(args.output_dir.glob("*.pt")) + [args.output_dir / "protocol.json", args.output_dir / "nnsight_parity.json"]
    write_json_atomic(args.output_dir / "DONE.json", {"complete": True, "outputs": [
        {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]})
    emit("nnsight_parity_complete", done_sha256=sha256(args.output_dir / "DONE.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "config", "bank", "diagnostic-dir", "reference-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json_atomic(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
