#!/usr/bin/env python3
"""Saved-data-only descriptive planner compensation check; no model/simulator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EPISODES = (0, 1, 4, 7, 10)
ARMS = ("unsteered", "identity", "subspace_up", "sham_up", "subspace_down", "sham_down")
CANDIDATE_SHA = "f4c295a148a22a9534cedd7d04b396f537bc8049842020886ba4f7fb0f1ca166"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def command_metrics(commands, baseline):
    commands, baseline = np.asarray(commands, float), np.asarray(baseline, float)
    if commands.shape != baseline.shape or commands.ndim != 2 or commands.shape[1] != 4:
        raise ValueError("Aligned [executed raw command,xyzw] arrays required")
    if not np.isfinite(commands).all() or not np.isfinite(baseline).all():
        raise ValueError("Nonfinite commands")
    def metrics(value):
        return {"z_sum": float(value[:, 2].sum()), "z_l1": float(np.abs(value[:, 2]).sum()),
                "z_l2": float(np.linalg.norm(value[:, 2])),
                "translation_chunk_frobenius": float(np.linalg.norm(value[:, :3])),
                "translation_net_xyz": value[:, :3].sum(0).tolist(),
                "translation_net_norm": float(np.linalg.norm(value[:, :3].sum(0)))}
    clipped, base_clipped = np.clip(commands, -1, 1), np.clip(baseline, -1, 1)
    raw, base = metrics(commands), metrics(baseline)
    return {"raw": raw, "baseline_raw": base,
            "raw_z_sum_change_vs_unsteered": raw["z_sum"] - base["z_sum"],
            "raw_z_l1_change_vs_unsteered": raw["z_l1"] - base["z_l1"],
            "raw_z_l2_change_vs_unsteered": raw["z_l2"] - base["z_l2"],
            "raw_z_delta_l2": float(np.linalg.norm(commands[:, 2] - baseline[:, 2])),
            "all_raw_command_delta_l2": float(np.linalg.norm(commands - baseline)),
            "unit_box_clipped_command_proxy": metrics(clipped),
            "unit_box_clipped_z_sum_change": float((clipped[:, 2] - base_clipped[:, 2]).sum()),
            "raw_values_outside_unit_box": int((np.abs(commands) > 1).sum()),
            "raw_z_values_outside_unit_box": int((np.abs(commands[:, 2]) > 1).sum()),
            "command_count": len(commands), "raw_commands": commands.tolist(),
            "interpretation": "Final native CEM returned commands submitted/executed during saved short fork; raw command units are not displacement or actuator controls. Unit-box clipping is only an explicitly recomputed proxy; internal scaling/clipping/applied actuator controls were not recorded."}


def opposition(left, right):
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("Finite comparisons required")
    if left == 0 or right == 0:
        return None
    return bool(left * right < 0)


def decode(readout, features):
    return ((features - readout["mean"]) / readout["scale"]) @ readout["coef"].T + readout["intercept"]


def pool_selected(prediction):
    visual = prediction["visual"].detach().cpu().double().numpy()
    if visual.shape[:2] != (7, 1) or visual.shape[-1] != 384:
        raise RuntimeError("Expected actual cached final CEM selected prediction [context+6,1,...,384]")
    return visual[1:].reshape(6, 1, -1, 384).mean(2)[:, 0]


def extract(directory, readout_path, output_dir):
    import torch
    torch.set_num_threads(1)
    receipt_path = directory / "DONE.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("complete") is not True or receipt["episode"] not in EPISODES
            or receipt["replan"] != 0 or receipt["candidate_sha256"] != CANDIDATE_SHA
            or receipt["beta"] != .5 or receipt["block"] != 3 or tuple(receipt["arms"]) != ARMS):
        raise RuntimeError("Only frozen five-start six-arm development diagnostic allowed")
    entries = {entry["path"]: entry for entry in receipt["outputs"]}
    if len(entries) != len(receipt["outputs"]):
        raise RuntimeError("Duplicate source receipt entries")
    def load(arm):
        path = directory / (arm + ".pt")
        if sha(path) != entries[path.name]["sha256"]:
            raise RuntimeError("Source SHA mismatch before tensor load: " + arm)
        return torch.load(path, map_location="cpu", weights_only=False)
    readout_done = json.loads((readout_path.parent / "DONE.json").read_text())
    readout_entry = [row for row in readout_done["outputs"] if row["path"] == readout_path.name]
    if (readout_done.get("complete") is not True or len(readout_entry) != 1
            or sha(readout_path) != readout_entry[0]["sha256"]):
        raise RuntimeError("Frozen discovery readout receipt mismatch")
    with np.load(readout_path) as values:
        readout = {key: values[key].copy() for key in ("mean", "scale", "coef", "intercept")}
    base = load("unsteered")
    base_latents = base["first_candidates"]["predicted_visual_pooled"].double().numpy()
    if base_latents.shape != (6, 300, 384):
        raise RuntimeError("Expected fixed first300-candidate prediction bank")
    base_xyz = decode(readout, base_latents)
    base_selected_xyz = decode(readout, pool_selected(base["selected_prediction"])) if base.get("selected_prediction") is not None else None
    base_commands = base["fork"]["actions"].double().numpy()
    base_states = np.asarray(base["fork"]["states"], float)
    rows, arrays = [], {}
    for arm in ARMS:
        result = base if arm == "unsteered" else load(arm)
        if not torch.equal(result["first_candidates"]["actions"], base["first_candidates"]["actions"]):
            raise RuntimeError("Initial candidate actions are not exactly paired")
        commands = result["fork"]["actions"].double().numpy()
        states = np.asarray(result["fork"]["states"], float)
        if states.shape != base_states.shape or len(commands) != len(base_commands) or len(commands) % 5:
            raise RuntimeError("Short-fork time alignment differs across arms")
        if not np.array_equal(states[:, -3:], base_states[:, -3:]):
            raise RuntimeError("Fixed physical goal changed")
        predicted = decode(readout, result["first_candidates"]["predicted_visual_pooled"].double().numpy())
        delta = predicted - base_xyz
        matched_step = len(commands) // 5
        if not 1 <= matched_step <= 6:
            raise RuntimeError("Executed prefix exceeds cached model horizon")
        action = command_metrics(commands, base_commands)
        realized_z = float(states[-1, 2] - base_states[-1, 2])
        mean_z = delta[:, :, 2].mean(1)
        selected = {"status": "missing", "reason": "No separately cached final selected-plan prediction; never substitute initial candidate-bank predictions"}
        if result.get("selected_prediction") is not None and base_selected_xyz is not None:
            selected_xyz = decode(readout, pool_selected(result["selected_prediction"]))
            selected_delta = selected_xyz - base_selected_xyz
            observed = states[4::5, :3]
            selected = {"status": "cached_final_CEM_selected_plan_probe_decoded",
                        "comparison": "Each edited arm's own optimized selected plan versus unsteered arm's own optimized selected plan; NOT a fixed-action comparison",
                        "decoded_hand_xyz_by_future_step": selected_xyz.tolist(),
                        "baseline_selected_decoded_hand_xyz_by_future_step": base_selected_xyz.tolist(),
                        "decoded_xyz_change_vs_baseline_selected_by_future_step": selected_delta.tolist(),
                        "decoded_z_change_at_executed_prefix_endpoint": float(selected_delta[matched_step - 1, 2]),
                        "actual_hand_xyz_by_executed_model_step": observed.tolist(),
                        "decoded_minus_actual_z_by_executed_model_step": (selected_xyz[:matched_step, 2] - observed[:, 2]).tolist(),
                        "unexecuted_predicted_future_steps": list(range(matched_step + 1, 7)),
                        "native_physical_xyz_prediction": "missing: native output is latent, decoded coordinates are frozen independent probe estimates"}
        if arm == "identity":
            if (not torch.equal(result["plan"], base["plan"]) or not np.array_equal(commands, base_commands)
                    or not np.array_equal(states, base_states) or np.count_nonzero(delta)):
                raise RuntimeError("Saved identity invariants failed")
        row = {"episode": receipt["episode"], "arm": arm, "arm_type": "semantic" if arm.startswith("subspace") else "sham" if arm.startswith("sham") else "control",
               "nominal_direction_label": "down" if arm.endswith("down") else "up" if arm.endswith("up") else None,
               "initial_same_candidate_probe_decoded_z_shift_mean_by_future_step": mean_z.tolist(),
               "initial_same_candidate_probe_decoded_z_shift_first_future": float(mean_z[0]),
               "initial_same_candidate_probe_decoded_z_shift_executed_horizon": float(mean_z[matched_step - 1]),
               "initial_same_candidate_probe_decoded_z_shift_full_horizon": float(mean_z[-1]),
               "initial_candidate_count": 300, "same_initial_candidate_actions_exact": True,
               "initial_bank_argmin": int(result["first_candidates"]["costs"].argmin()),
               "initial_bank_mean_native_terminal_cost_change": float((result["first_candidates"]["costs"] - base["first_candidates"]["costs"]).double().mean()),
               "final_selected_commands": action,
               "normalized_returned_plan_shape": list(result["plan"].shape),
               "executed_model_steps": matched_step, "raw_fork_steps": len(commands),
               "realized_endpoint_z_shift_vs_unsteered_m": realized_z,
               "realized_z_shift_by_executed_model_step": (states[4::5, 2] - base_states[4::5, 2]).tolist(),
               "selected_plan_prediction": selected,
               "actual_applied_actuator_controls": {"status": "not_saved"}}
        row["signed_opposition"] = {
            name: {"raw_command_z_change_opposes_initial_predicted_z_shift": opposition(float(value), action["raw_z_sum_change_vs_unsteered"]),
                   "unit_box_clipped_proxy_opposes_initial_predicted_z_shift": opposition(float(value), action["unit_box_clipped_z_sum_change"]),
                   "realized_z_opposes_initial_predicted_z_shift": opposition(float(value), realized_z)}
            for name, value in (("first_future", mean_z[0]), ("executed_horizon", mean_z[matched_step - 1]), ("full_horizon", mean_z[-1]))}
        rows.append(row)
        arrays[arm + "__initial_candidate_decoded_z_shift"] = delta[:, :, 2]
        arrays[arm + "__raw_command_delta"] = commands - base_commands
        print(json.dumps({"event": "compensation_arm_extracted", "episode": receipt["episode"], "arm": arm}), flush=True)
    np.savez_compressed(output_dir / "candidate_and_command_arrays.npz", **arrays)
    write(output_dir / "episode_compensation.json", {"complete": True, "episode": receipt["episode"], "rows": rows,
          "candidate_sha256": CANDIDATE_SHA, "source_done_sha256": sha(receipt_path), "source_done": str(receipt_path),
          "readout_sha256": sha(readout_path), "script_sha256": sha(__file__),
          "source_tensor_sha256": {arm: entries[arm + ".pt"]["sha256"] for arm in ARMS},
          "new_model_forwards": 0, "new_simulator_steps": 0, "identity_exact": True})


def align_displacements(selected_xyz, actual_xyz, observed_start):
    selected_xyz, actual_xyz, observed_start = map(lambda value: np.asarray(value, float), (selected_xyz, actual_xyz, observed_start))
    if selected_xyz.shape != (6, 3) or actual_xyz.ndim != 2 or actual_xyz.shape[1] != 3 or observed_start.shape != (3,):
        raise ValueError("Expected six predicted states, executed model-step states, and actual observed startXYZ")
    if not 1 <= len(actual_xyz) <= 6:
        raise ValueError("Invalid executed horizon")
    return {"observed_start_hand_xyz_m": observed_start.tolist(),
            "predicted_displacement_from_observed_start_xyz_m": (selected_xyz - observed_start).tolist(),
            "actual_displacement_from_observed_start_xyz_m": (actual_xyz - observed_start).tolist(),
            "actual_raw_action_indices": list(range(5, 5 * len(actual_xyz) + 1, 5)),
            "predicted_raw_action_indices": [5, 10, 15, 20, 25, 30],
            "first_five_action_predicted_z_displacement_m": float(selected_xyz[0, 2] - observed_start[2]),
            "first_five_action_actual_z_displacement_m": float(actual_xyz[0, 2] - observed_start[2]),
            "alignment": "Subtract identical observed current handXYZ after reset warmup; predicted/actual states at raw actions5,10,15 align, later predictions unobserved"}


def summarize(inputs, output_dir, reference_extractions=None):
    episodes = []
    for path in inputs:
        receipt = json.loads((path.parent / "DONE.json").read_text())
        entries = [row for row in receipt["outputs"] if row["path"] == path.name]
        if receipt.get("complete") is not True or len(entries) != 1 or sha(path) != entries[0]["sha256"]:
            raise RuntimeError("Extraction receipt mismatch")
        episodes.append(json.loads(path.read_text()))
    episodes.sort(key=lambda x: x["episode"])
    if tuple(row["episode"] for row in episodes) != EPISODES:
        raise RuntimeError("All five predeclared development starts required")
    rows = [row for episode in episodes for row in episode["rows"]]
    reference_sources = []
    if reference_extractions:
        references = {}
        for path in reference_extractions:
            receipt = json.loads((path.parent / "DONE.json").read_text())
            entries = [row for row in receipt["outputs"] if row["path"] == path.name]
            if receipt.get("complete") is not True or len(entries) != 1 or sha(path) != entries[0]["sha256"]:
                raise RuntimeError("Observed-start extraction SHA mismatch")
            reference = json.loads(path.read_text())
            expected = next(episode for episode in episodes if episode["episode"] == reference["episode"])
            if reference["diagnostic_receipt_sha256"] != expected["source_done_sha256"]:
                raise RuntimeError("Observed-start source does not match causal diagnostic")
            references[reference["episode"]] = reference
            reference_sources.append({"path": str(path), "sha256": sha(path)})
        if set(references) != set(EPISODES):
            raise RuntimeError("Observed-start alignment requires allfive exact episode references")
        for row in rows:
            old = next(value for value in references[row["episode"]]["rows"] if value["arm"] == row["arm"])
            if row["realized_endpoint_z_shift_vs_unsteered_m"] != old["endpoint_effect_vs_unsteered_xyz_m"][2]:
                raise RuntimeError("Realized endpoint does not match independent extraction")
            selected = row["selected_plan_prediction"]
            if selected["status"] != "missing":
                selected.update(align_displacements(selected["decoded_hand_xyz_by_future_step"],
                                                   selected["actual_hand_xyz_by_executed_model_step"], old["start_hand_xyz_m"]))
    counts = {}
    for arm in ARMS[2:]:
        selected = [row for row in rows if row["arm"] == arm]
        counts[arm] = {}
        for horizon in ("first_future", "executed_horizon", "full_horizon"):
            counts[arm][horizon] = {}
            for metric in selected[0]["signed_opposition"][horizon]:
                values = [row["signed_opposition"][horizon][metric] for row in selected]
                counts[arm][horizon][metric] = {"opposed": values.count(True), "aligned": values.count(False), "zero_unclassified": values.count(None), "N_starts": 5}
    write(output_dir / "planner_compensation.json", {"complete": True, "rows": rows, "opposition_counts": counts,
          "episode_ids": list(EPISODES), "N_independent_development_starts": 5, "candidate_count_is_not_independent_N": True,
          "source_extractions": [{"path": str(path), "sha256": sha(path)} for path in inputs],
          "observed_start_reference_extractions": reference_sources,
          "hypothesis": "Predicted positive motion bias may be followed by the optimizer choosing opposite raw commands; sign opposition is descriptive compatibility, not proof of a compensation mechanism",
          "limitations": ["Frozen discovery probe errors can affect predicted-shift signs", "Initial300 candidate effects precede optimization and are not final selected-plan predictions", "Edited arms optimize different final plans; their own-plan prediction differences mix representation and selected-action effects", "Raw command sums are not clipped/scaled actuator controls or physical displacement", "Unit-box clipping is only a reconstruction proxy; applied controls not saved", "Five starts and paired directions/shams do not support independent-candidate inference", "No fixed-action prediction-error improvement or mediation test was run", "No new gates, model passes, simulator steps, or held tensors"],
          "new_model_forwards": 0, "new_simulator_steps": 0, "script_sha256": sha(__file__)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "summarize"))
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--readout", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--reference-extractions", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == "extract":
            extract(args.episode_dir, args.readout, args.output_dir)
        else:
            summarize(args.inputs, args.output_dir, args.reference_extractions)
        outputs = sorted(path for path in args.output_dir.iterdir() if path.is_file())
        write(args.output_dir / "DONE.json", {"complete": True, "outputs": [{"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in outputs]})
    except Exception as exc:
        write(args.output_dir / "FAILED.json", {"complete": False, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
