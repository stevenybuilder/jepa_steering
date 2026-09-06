#!/usr/bin/env python3
"""Supplement one completed development diagnostic with fixed-candidate traces.

Reuse the recorded 300 initial H6 candidates for six arms. No CEM optimization,
new simulator action fork, operator fitting, or held-out trajectory is performed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from causal_planner_forks import (
    checked_max_error, close_env, emit, load_development_bank, reconstruct,
    setup_cfg, visual_pooled,
)
from collect_on_policy_bank import block_list
from model_loader import load_headless_metaworld
from protocol import file_sha256, write_json_atomic
from screen_pooled import load_discovery
from subspace_control import FirstStepSubspacePatch


def activation_statistics(before, after, requested, basis, scale):
    """Measure complement residuals in both explicitly defined metrics."""
    realized = after - before
    raw_basis = torch.linalg.qr(scale[:, None] * basis, mode="reduced").Q
    result = {"before_pooled": before, "after_pooled": after,
              "requested_delta": requested, "realized_delta": realized}
    for name, delta in (("requested", requested), ("realized", realized)):
        standardized = delta / scale
        outside_standardized = standardized - (standardized @ basis) @ basis.T
        outside_raw = delta - (delta @ raw_basis) @ raw_basis.T
        result.update({
            f"{name}_raw_l2": delta.norm(dim=-1),
            f"{name}_standardized_l2": standardized.norm(dim=-1),
            f"{name}_raw_orthogonal_l2": outside_raw.norm(dim=-1),
            f"{name}_standardized_orthogonal_l2": outside_standardized.norm(dim=-1),
        })
    return {key: value.detach().cpu() for key, value in result.items()}


def ranks_by_step(costs):
    if costs.ndim != 2 or not torch.isfinite(costs).all():
        raise RuntimeError("Expected finite [future step,candidate] costs")
    return torch.from_numpy(rankdata(costs.cpu().numpy(), axis=1, method="average"))


def checked_artifact(directory, receipt, name):
    matches = [row for row in receipt["outputs"] if row["path"] == name]
    if not receipt.get("complete") or len(matches) != 1:
        raise RuntimeError(f"Missing completed artifact receipt: {name}")
    path = directory / name
    if file_sha256(path) != matches[0]["sha256"]:
        raise RuntimeError(f"Artifact hash mismatch: {name}")
    return torch.load(path, map_location="cpu", weights_only=False)


def require_exact(actual, expected, label):
    if not torch.equal(actual, expected):
        detail = "shape mismatch" if actual.shape != expected.shape else f"max absolute error {float((actual.double() - expected.double()).abs().max())}"
        raise RuntimeError(f"{label} is not exactly identical: {detail}")


def compare_float32_replay(actual, expected, label):
    """Frozen cross-process bound; within-run identity still uses require_exact."""
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{label} replay shape or finiteness mismatch")
    error = float((actual.double() - expected.double()).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"{label} replay max absolute error {error} > frozen 1e-6 bound")
    return {"max_absolute_error": error, "bound": 1e-6,
            "actual_dtype": str(actual.dtype), "cached_dtype": str(expected.dtype)}


@torch.no_grad()
def run(args):
    bank, bank_sha = load_development_bank(args.bank)
    receipt = json.loads((args.diagnostic_dir / "DONE.json").read_text())
    if (not receipt.get("complete") or receipt["episode"] != 0 or receipt["replan"] != 0
            or bank["episode"] != 0 or receipt["bank_sha256"] != bank_sha
            or receipt["beta"] != 0.5 or receipt["block"] != 3):
        raise RuntimeError("Expected completed episode-0/replan-0 block-3 beta-0.5 diagnostic")
    candidate_receipt = json.loads((args.candidate.parent / "DONE.json").read_text())
    candidate_sha = file_sha256(args.candidate)
    if (not candidate_receipt.get("complete") or candidate_sha != candidate_receipt["sha256"]
            or candidate_sha != receipt["candidate_sha256"]):
        raise RuntimeError("Frozen candidate hash mismatch")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    counterfactuals = checked_artifact(args.diagnostic_dir, receipt, "counterfactuals.pt")
    baseline = checked_artifact(args.diagnostic_dir, receipt, "unsteered.pt")
    fixed_actions = baseline["first_candidates"]["actions"].clone()
    if fixed_actions.shape != (6, 300, 20):
        raise RuntimeError("Expected the recorded native [H6,300,20] candidates")

    discovery = load_discovery(args.capture_dirs)
    natural = torch.from_numpy(discovery["features"]["predictor"][:, 3]).float().clone()
    trajectories = set(zip(discovery["seeds"].tolist(), discovery["episodes"].tolist()))
    if len(trajectories) != 150 or natural.shape != (2850, 400):
        raise RuntimeError("Expected all 150 discovery trajectories at block 3")
    checked_max_error(natural.double().mean(0).float(), candidate["mean"], "discovery metric mean", 1e-5)
    natural_scale = natural.double().std(0, correction=0).float()
    natural_scale[natural_scale < 1e-6] = 1.0
    checked_max_error(natural_scale, candidate["scale"], "discovery metric scale", 1e-5)
    torch.save({"block_pooled": natural, "seeds": torch.from_numpy(discovery["seeds"]),
                "episodes": torch.from_numpy(discovery["episodes"]),
                "mean": candidate["mean"], "scale": candidate["scale"], "basis": candidate["basis"],
                "sources": discovery["sources"], "split": "discovery"}, args.output_dir / "natural_reference.pt")
    del discovery, natural

    wm, preprocessor, provenance = load_headless_metaworld(args.repo)
    for parameter in wm.parameters():
        parameter.requires_grad_(False)
    cfg = setup_cfg(args.config, args.output_dir, wm)
    _, blocks = block_list(wm.model.predictor, ("predictor_blocks", "blocks"), 6, "predictor")
    block = blocks[3]
    basis, scale = candidate["basis"].cuda(), candidate["scale"].cuda()
    protocol = {
        "complete": False, "scope": "one development state, six arms, identical recorded candidates; candidates are not independent episodes",
        "episode": 0, "replan": 0, "block": 3, "imagined_edit_step": 0, "rank": basis.shape[1], "beta": 0.5,
        "candidate_count": 300, "future_model_steps": 6, "new_cem_optimizations": 0, "new_physical_forks": 0,
        "arms": receipt["arms"], "bank_sha256": bank_sha, "candidate_sha256": candidate_sha,
        "diagnostic_receipt_sha256": file_sha256(args.diagnostic_dir / "DONE.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "patch_script_sha256": file_sha256(Path(__file__).with_name("subspace_control.py")),
        "model": provenance, "future_layout": "[future step 1..6,candidate,...]; observed context excluded",
        "trace_dtype": "float32", "rank_convention": "1-based average rank for ties, lower cost is better",
        "raw_complement_metric": "Euclidean complement of QR(diag(scale) @ standardized_basis)",
        "standardized_complement_metric": "Euclidean complement of saved orthonormal basis after dividing by saved scale",
        "natural_reference": {"trajectories": 150, "states": 2850, "split": "discovery", "sources": candidate_receipt["sources"]},
        "cross_process_numerical_policy": {"max_absolute_error": 1e-6, "applies_to": ["costs", "pooled_visual", "proprio"],
            "unsteered_cost_ranks_and_argmin": "exact", "within_run_identity_complete_traces": "bit exact",
            "prior_failed_cross_process_max_cost_error": 2.384185791015625e-7,
            "scope": "float32 replay consistency only; no effect or specificity acceptance criterion changed"},
        "remaining_controls": ["unrelated-site", "time-shift", "held-out confirmation", "full-episode efficacy"],
    }
    write_json_atomic(args.output_dir / "protocol.json", protocol)
    rows, identity_checks, baseline_prediction, baseline_activation = [], {}, None, None
    for arm in receipt["arms"]:
        started = time.monotonic()
        cached = baseline if arm == "unsteered" else checked_artifact(args.diagnostic_dir, receipt, f"{arm}.pt")
        require_exact(cached["first_candidates"]["actions"], fixed_actions, f"{arm} cached actions")
        env, agent, z, replay_checks = reconstruct(cfg, wm, preprocessor, bank, 0)
        if agent.objective.sum_all_diffs:
            raise RuntimeError("Expected native terminal-distance objective, not cumulative objective")
        actions = fixed_actions.cuda()
        donor = counterfactuals["donors"]["down" if arm.endswith("down") else "up"].cuda()
        patch = FirstStepSubspacePatch(block, wm.unroll, basis, scale, donor,
                                      beta=0.0 if arm in ("identity", "unsteered") else 0.5,
                                      sham=arm.startswith("sham"))
        emit("fixed_candidates_unroll_started", arm=arm, horizon=6, candidates=300)
        if arm == "unsteered":
            captured = []

            def capture(_module, _args, output):
                if not captured:
                    if output.shape != (300, 256, 400):
                        raise RuntimeError(f"Unexpected block capture shape {output.shape}")
                    captured.append(output.detach().float().mean(1))

            handle = block.register_forward_hook(capture)
            try:
                prediction = wm.unroll(z, act_suffix=actions)
            finally:
                handle.remove()
            before = after = captured[0]
            requested = torch.zeros_like(before)
        else:
            with patch:
                prediction = patch.unroll(z, act_suffix=actions)
            if patch.calls != 1 or patch.step != 6:
                raise RuntimeError("Patch must edit only the first of six imagined steps")
            before = patch.first_activation["before_pooled"].cuda()
            after = patch.first_activation["after_pooled"].cuda()
            requested = torch.zeros_like(before) if patch.first_edit is None else patch.first_edit.cuda()
        require_exact(actions.cpu(), fixed_actions, f"{arm} unmodified actions")
        if cached["edit"] is not None:
            require_exact(requested.cpu(), cached["edit"], f"{arm} original diagnostic edit")
        pooled = visual_pooled(prediction, actions).cpu()
        costs = agent.objective(prediction, actions).cpu()
        step_costs = agent.objective(prediction, actions, keepdims=True)[1:].cpu()
        if step_costs.shape != (6, 300):
            raise RuntimeError(f"Unexpected cost trace {step_costs.shape}")
        cross_process_checks = {
            "costs": compare_float32_replay(costs, cached["first_candidates"]["costs"], f"{arm} costs"),
            "pooled_visual": compare_float32_replay(pooled, cached["first_candidates"]["predicted_visual_pooled"], f"{arm} pooled visual"),
            "proprio": compare_float32_replay(prediction["proprio"].cpu(), cached["first_candidates"]["predicted_proprio"], f"{arm} proprio"),
        }
        if arm == "unsteered":
            require_exact(ranks_by_step(costs.unsqueeze(0)), ranks_by_step(cached["first_candidates"]["costs"].unsqueeze(0)), "unsteered original cost ranks")
            require_exact(costs.argmin(), cached["first_candidates"]["costs"].argmin(), "unsteered original cost argmin")
        require_exact(step_costs[-1], costs, f"{arm} final-step costs")
        future = {name: prediction[name][1:].detach().cpu() for name in ("visual", "proprio")}
        if any(not torch.isfinite(value).all() for value in future.values()):
            raise RuntimeError("Non-finite future trace")
        support = activation_statistics(before, after, requested, basis, scale)
        if arm == "unsteered":
            baseline_prediction = future
            baseline_activation = support["before_pooled"]
        else:
            require_exact(support["before_pooled"], baseline_activation, f"{arm} identical pre-edit activation")
        if arm == "identity":
            for name in ("visual", "proprio"):
                require_exact(future[name], baseline_prediction[name], f"identity complete future {name}")
                identity_checks[f"future_{name}_exact"] = True
            require_exact(support["before_pooled"], support["after_pooled"], "identity activation")
            identity_checks.update({"activation_exact": True, "costs_exact": True, "actions_exact": True})
            baseline_prediction = None
        result = {"arm": arm, "actions": fixed_actions, "prediction": future,
                  "predicted_visual_pooled": pooled, "costs_by_step": step_costs,
                  "ranks_by_step": ranks_by_step(step_costs), "terminal_costs": costs,
                  "activation_support": support, "replay_checks": replay_checks,
                  "cross_process_checks": cross_process_checks, "first_step_hook_calls": patch.calls}
        torch.save(result, args.output_dir / f"{arm}.pt")
        close_env(env)
        rows.append({"arm": arm, "seconds": time.monotonic() - started,
                     "cross_process_checks": cross_process_checks,
                     "mean_requested_raw_edit_l2": float(support["requested_raw_l2"].mean()),
                     "max_realized_standardized_orthogonal_l2": float(support["realized_standardized_orthogonal_l2"].max())})
        write_json_atomic(args.output_dir / "progress.json", {"complete": False, "rows": rows})
        emit("support_arm_complete", **rows[-1])
        del prediction, future, result, cached, agent, z, env
    write_json_atomic(args.output_dir / "DONE.json", {
        **protocol, "complete": True, "rows": rows, "identity_checks": identity_checks,
        "outputs": [{"path": p.name, "sha256": file_sha256(p)} for p in sorted(args.output_dir.glob("*.pt"))],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "config", "bank", "candidate", "diagnostic-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--capture-dirs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json_atomic(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
