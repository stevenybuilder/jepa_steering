#!/usr/bin/env python3
"""TRAIN-source-grouped Push coordinate/time readouts; no confirmation or planning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import Ridge
from screen_coordinate_time import fit_models, predict_models, per_episode_ci, residual_basis
from complete_cached_geometry import sha256, write_json

FRAMES = ("world", "gripper_relative", "object_relative")
ALPHAS = (.01, 1.)
CHECKPOINT = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
PROTOCOL = {
    "version": 1, "task": "pusht", "site": "predictor_block3", "input": "all400 pooled residual channels",
    "frames": FRAMES, "models": ["linear", "rbf"], "alphas_each": ALPHAS,
    "linear_kernel": "standardized dot product /400", "rbf_gamma": "1/(2*400)", "pca": False,
    "fit_sources": [0, 99], "validation_sources": [100, 124], "confirmation_access": False,
    "selection": "five discovery-only source-group folds, source_id modulo5; all6 transitions stay together",
    "candidate_selection": "frame with minimum discovery-CV LINEAR common world endpoint MSE",
    "candidate_ranks": [1, 2, 8], "independent_unit": "whole TRAIN source trajectory",
    "time_target": "actual input raw frame index0,5,...25; not appended to activation inputs",
    "imagined_horizon": "constant1 model step; no multi-horizon claim",
    "frame_inverse_uses_evaluation_anchors": True,
    "object_frame": "actual current block center and rotation constructed from true simulator sin/cos(theta)",
    "gripper_frame": "translation by actual current agentXY; no invented agent orientation",
    "goal": "absent in TRAIN capture; not invented from finalclipframe",
    "motion_speed": "net physical displacement over5 rawsteps /5, not path speed or pixels/second",
    "limits": ["readout encodings sin/cos and coordinate transforms are analyst targets, not established native frames",
               "nonlinear readout gain does not identify an intrinsic manifold",
               "relative-frame reconstruction uses privileged true current pose anchors",
               "real input time may be inferred from progress; not evidence of a native clock",
               "no causal intervention or held scripted performance evaluation"]}


def selected(entry, split):
    bounds = {"fit": (0, 100), "validation": (100, 125)}
    if split not in bounds: raise ValueError("Confirmation access forbidden")
    lo, hi = bounds[split]
    source = int(entry["source_id"])
    return lo <= source < hi and entry["split"] == split


def frame_targets(current, following, current_theta, next_theta, raw_step):
    current, following = np.asarray(current, float), np.asarray(following, float)
    theta = np.asarray(current_theta, float)
    n = len(current)
    identity = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    rotation = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)], -1).reshape(n, 2, 2)
    origins = {"world": np.zeros((n, 2)), "gripper_relative": current[:, :2], "object_relative": current[:, 2:4]}
    rotations = {"world": identity, "gripper_relative": identity, "object_relative": rotation}
    arrays, columns, start = [], {}, 0
    for name in FRAMES:
        local = np.einsum("npi,nij->npj", following.reshape(n, 2, 2)-origins[name][:, None], rotations[name]).reshape(n, 4)
        arrays.append(local); columns[name] = slice(start, start+4); start += 4
    extra = {"block_angle_sin_cos": np.column_stack([np.sin(next_theta), np.cos(next_theta)]),
             "physical_motion_world": following-current,
             "physical_net_speed_per_raw_step": np.linalg.norm((following-current).reshape(n, 2, 2), axis=-1)/5.,
             "block_angular_motion_wrapped": np.arctan2(np.sin(next_theta-theta), np.cos(next_theta-theta))[:, None],
             "observed_episode_raw_step": np.asarray(raw_step).reshape(n, 1)}
    for key, value in extra.items():
        arrays.append(value); columns[key] = slice(start, start+value.shape[1]); start += value.shape[1]
    return np.concatenate(arrays, axis=1), columns, origins, rotations


def world_prediction(data, name, local):
    n = len(local)
    return (np.einsum("npi,nji->npj", local.reshape(n, 2, 2), data["rotations"][name]) +
            data["origins"][name][:, None]).reshape(n, 4)


def load_split(directory, split, frozen=None):
    if split == "validation" and (frozen is None or not frozen.is_file()): raise ValueError("Freeze before validation load")
    receipt = json.loads((directory/"DONE.json").read_text())
    if not receipt.get("complete") or receipt["checkpoint_sha256"] != CHECKPOINT: raise ValueError("Wrong/incomplete capture")
    # Select by receipt BEFORE any tensor/hash access, and never select confirmation.
    entries = [row for row in receipt["outputs"] if selected(row, split)]
    expected = list(range(100)) if split == "fit" else list(range(100, 125))
    if sorted(row["source_id"] for row in entries) != expected: raise ValueError("Missing/duplicate source")
    import torch
    values = {key: [] for key in ("x", "current", "following", "theta", "next_theta", "raw_step", "episode")}
    sources = []
    for row in entries:
        path = directory/row["path"]
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != row["sha256"]: raise ValueError("Unsafe/mismatched source")
        bank = torch.load(path, map_location="cpu", weights_only=True)
        meta = bank["meta"]
        if meta["split"] != split or meta["source_id"] != row["source_id"] or meta["checkpoint_sha256"] != CHECKPOINT: raise ValueError("Capture metadata mismatch")
        indices = np.asarray(meta["observed_raw_indices"])
        states = bank["states"].numpy().astype(float)
        x = bank["predictor_p3_pooled"].numpy().astype(float)
        if x.shape != (6, 400) or states.shape != (7, 7) or not np.array_equal(indices, np.arange(0, 31, 5)): raise ValueError("Misaligned source")
        if not np.allclose(bank["q_next"].numpy()[:, :4], states[1:, :4], atol=1e-5): raise ValueError("Wrong physical next labels")
        source_values = dict(x=x, current=states[:-1, :4], following=states[1:, :4], theta=states[:-1, 4],
                             next_theta=states[1:, 4], raw_step=indices[:-1], episode=np.full(6, row["source_id"]))
        for key, value in source_values.items():
            if not np.isfinite(value).all(): raise ValueError("Nonfinite capture")
            values[key].append(value)
        sources.append({"source_id": row["source_id"], "split": split, "path": str(path), "sha256": row["sha256"]})
    data = {key: np.concatenate(value) for key, value in values.items()}
    y, cols, origins, rotations = frame_targets(data["current"], data["following"], data["theta"], data["next_theta"], data["raw_step"])
    return dict(**data, y=y, columns=cols, origins=origins, rotations=rotations, sources=sources)


def choose(data, predictions):
    return {variable: {family: min((key for key in predictions if key.startswith(family+":")),
             key=lambda key: np.square(predictions[key][:, cols]-data["y"][:, cols]).mean()) for family in ("linear", "rbf")}
             for variable, cols in data["columns"].items()}


def mse(y, prediction): return np.square(y-prediction).mean(1)


def evaluate(data, predictions, chosen, mean_baseline, time_baseline):
    report = {}
    for variable, cols in data["columns"].items():
        truth = data["y"][:, cols]
        row, errors = {}, {}
        for family, key in chosen[variable].items():
            predicted = predictions[key][:, cols]
            error = mse(truth, predicted)
            errors[family] = error
            denominator = float(np.square(truth-truth.mean(0)).sum())
            row[family] = {"model": key, "rmse": float(np.sqrt(error.mean())),
                           "r2": float(1-np.square(predicted-truth).sum()/denominator) if denominator > 1e-15 else None}
            if variable in FRAMES:
                world = world_prediction(data, variable, predicted)
                row[family].update(world_rmse_pixels=float(np.sqrt(mse(data["following"], world).mean())),
                    agent_world_rmse_pixels=float(np.sqrt(mse(data["following"][:, :2], world[:, :2]).mean())),
                    block_world_rmse_pixels=float(np.sqrt(mse(data["following"][:, 2:], world[:, 2:]).mean())))
        row["nonlinear_minus_linear_mse"] = per_episode_ci(errors["rbf"]-errors["linear"], data["episode"])
        row["training_mean_baseline_rmse"] = float(np.sqrt(mse(truth, mean_baseline[:, cols]).mean()))
        row["six_bin_time_baseline_rmse"] = float(np.sqrt(mse(truth, time_baseline[:, cols]).mean()))
        row["time_baseline_circular_for_time_target"] = variable == "observed_episode_raw_step"
        if variable in FRAMES:
            row["mean_baseline_world_rmse_pixels"] = float(np.sqrt(mse(data["following"], world_prediction(data, variable, mean_baseline[:, cols])).mean()))
            row["time_baseline_world_rmse_pixels"] = float(np.sqrt(mse(data["following"], world_prediction(data, variable, time_baseline[:, cols])).mean()))
            row["physical_persistence_world_rmse_pixels"] = float(np.sqrt(mse(data["following"], data["current"]).mean()))
        report[variable] = row
    return report


def freeze_candidate(data, bundle, chosen, cv, directory):
    frame = min(FRAMES, key=lambda name: cv[name]["linear"]["world_rmse_pixels"])
    key = chosen[frame]["linear"]
    cols = data["columns"][frame]
    basis = residual_basis(data["x"], data["y"][:, cols], float(key.split(":")[1]))
    if basis.shape[1] < 8: raise ValueError("Requested raw rank8 not identified; no fabricated completion")
    model = bundle["models"][key]
    coef = model.coef_[cols]*bundle["target_scale"][cols, None]/bundle["scale"][None, :]
    intercept = model.intercept_[cols]*bundle["target_scale"][cols]+bundle["target_mean"][cols]-coef@bundle["mean"]
    radius = .005*float(np.sqrt(np.sum(data["x"]**2, axis=1).mean()))
    path = directory/"residual_candidate.npz"
    np.savez_compressed(path, **{f"basis_raw_rank{k}": basis[:, :k] for k in (1, 2, 8)},
        coordinate_coef_raw=coef, coordinate_intercept_raw=intercept,
        raw_feature_mean=bundle["mean"], raw_feature_scale=bundle["scale"],
        donor_p3=data["x"], donor_target=data["y"][:, cols], donor_current_hand=data["current"][:, :2],
        donor_current_agent=data["current"][:, :2], donor_current_block=data["current"][:, 2:],
        donor_block_sin_cos=np.column_stack([np.sin(data["theta"]), np.cos(data["theta"])]),
        donor_current_physical=data["current"], donor_next_physical=data["following"],
        donor_real_raw_step=data["raw_step"], donor_episode=data["episode"], donor_source_id=data["episode"],
        cap_radius=radius, target_name=np.asarray(frame), max_rank=np.asarray(basis.shape[1]))
    return {"path": path.name, "sha256": sha256(path), "target": frame, "rank": int(basis.shape[1]),
        "selection": "minimum fit-only grouped-CV LINEAR world endpoint error; validation unopened",
        "cap_radius": radius, "cap_definition": ".005*RMS norm of fit pooledP3 raw Euclidean space",
        "intervention": "donor subspace replacement; not probe-label optimization", "native_coordinates_established": False,
        "donor_sources": 100, "donor_transitions": 600, "goal_available": False, "simulator_seed_available": False,
        "schema_note": "same raw basis/coefficient/donor keys as Reach; hand alias means agentXY, endpoints4D; no invented donor_goal/donor_seed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"protocol.json", PROTOCOL)
    data = load_split(args.capture_dir, "fit")
    predictions = {f"{family}:{alpha}": np.empty_like(data["y"]) for family in ("linear", "rbf") for alpha in ALPHAS}
    mean_baseline, time_baseline = np.empty_like(data["y"]), np.empty_like(data["y"])
    time_features = np.eye(6)[(data["raw_step"]//5).astype(int)]
    for fold in range(5):
        test = data["episode"]%5 == fold
        train = ~test
        bundle = fit_models(data["x"][train], data["y"][train])
        for key, value in predict_models(bundle, data["x"][test]).items(): predictions[key][test] = value
        mean_baseline[test] = data["y"][train].mean(0)
        time_baseline[test] = Ridge(alpha=1.).fit(time_features[train], data["y"][train]).predict(time_features[test])
        print(json.dumps({"event": "fit_source_fold_complete", "fold": fold}), flush=True)
    chosen = choose(data, predictions)
    cv = evaluate(data, predictions, chosen, mean_baseline, time_baseline)
    bundle = fit_models(data["x"], data["y"])
    time_model = Ridge(alpha=1.).fit(time_features, data["y"])
    import joblib
    joblib.dump({"bundle": bundle, "time_only": time_model}, args.output_dir/"readouts.joblib")
    candidate = freeze_candidate(data, bundle, chosen, cv, args.output_dir)
    frozen = {"chosen_models": chosen, "candidate": candidate, "readouts_sha256": sha256(args.output_dir/"readouts.joblib"),
              "discovery_results": cv, "discovery_sources": data["sources"], "validation_opened": False, "confirmation_opened": False,
              "common_source_sha256": sha256(Path(__file__).with_name("screen_coordinate_time.py"))}
    write_json(args.output_dir/"FROZEN.json", frozen)
    print(json.dumps({"event": "candidate_frozen", **candidate}), flush=True)
    val = load_split(args.capture_dir, "validation", args.output_dir/"FROZEN.json")
    vp = predict_models(bundle, val["x"])
    vm = np.broadcast_to(data["y"].mean(0), val["y"].shape)
    vt = time_model.predict(np.eye(6)[(val["raw_step"]//5).astype(int)])
    result = {"complete": True, "discovery": cv, "validation": evaluate(val, vp, chosen, vm, vt),
        "candidate": candidate, "protocol": PROTOCOL, "discovery_episodes": 100, "validation_episodes": 25,
        "discovery_transitions": 600, "validation_transitions": 150, "confirmation_opened": False,
        "native_coordinate_claim": False, "sources": val["sources"], "seconds": time.monotonic()-started,
        "goal_scope": "No goal in TRAIN capture; no goal-relative target. Original10 development Push prepared inputs retain true goals for later authorized multi-horizon capture.",
        "physical_endpoint_units": "native512arena pixels", "speed_units": "net pixels per raw step", "time_units": "raw steps"}
    np.savez_compressed(args.output_dir/"validation_predictions.npz", **vp, targets=val["y"],
        episode=val["episode"], raw_step=val["raw_step"], following_world=val["following"],
        mean_baseline=vm, time_baseline=vt, target_columns=np.asarray(json.dumps({k:[v.start,v.stop] for k,v in val["columns"].items()})))
    write_json(args.output_dir/"coordinate_time_results.json", result)
    write_json(args.output_dir/"DONE.json", {"complete": True, "confirmation_opened": False,
        "script_sha256": sha256(__file__), "outputs": [{"path": p.name, "sha256": sha256(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file()]})
    print(json.dumps({"event": "validation_complete", "seconds": result["seconds"], "selected_frame": candidate["target"]}), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2): main()
