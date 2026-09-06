#!/usr/bin/env python3
"""Post-hoc coordinate-transform control; no refitting or confirmation access."""
import argparse
import json
from pathlib import Path

import numpy as np

from complete_cached_geometry import sha256, to_local, write_json
from screen_coordinate_time import load_split, per_episode_ci


def transform_control(world_prediction, current, following, rotation):
    world_prediction, current, following, rotation = (
        np.asarray(value, dtype=np.float64) for value in (world_prediction, current, following, rotation)
    )
    direct_error = np.square(world_prediction-following).mean(1)
    converted = to_local(world_prediction-current, rotation)
    target = to_local(following-current, rotation)
    rotated_error = np.square(converted-target).mean(1)
    if not np.allclose(direct_error, rotated_error, atol=1e-12, rtol=1e-7):
        raise ValueError("Coordinate transform did not preserve Euclidean prediction error")
    return converted, target, rotated_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.frozen_dir/"DONE.json").read_text())
    for item in receipt["outputs"]:
        if sha256(args.frozen_dir/item["path"]) != item["sha256"]:
            raise ValueError("Frozen readout output changed")
    frozen = json.loads((args.frozen_dir/"FROZEN.json").read_text())
    predictions = np.load(args.frozen_dir/"validation_predictions.npz")
    data = load_split(args.capture_dirs, "validation")
    np.testing.assert_array_equal(predictions["seed"], data["seed"])
    np.testing.assert_array_equal(predictions["episode"], data["episode"])
    np.testing.assert_allclose(predictions["targets"], data["y"], atol=0, rtol=0)
    world_key = frozen["chosen_models"]["world"]["linear"]
    linear_key = frozen["chosen_models"]["goal_aligned"]["linear"]
    nonlinear_key = frozen["chosen_models"]["goal_aligned"]["rbf"]
    world = predictions[world_key][:, data["columns"]["world"]]
    converted, target, error = transform_control(world, data["current"], data["following"], data["rotations"]["goal_aligned"])
    learned_linear = predictions[linear_key][:, data["columns"]["goal_aligned"]]
    learned_rbf = predictions[nonlinear_key][:, data["columns"]["goal_aligned"]]
    groups = data["seed"]*10000+data["episode"]
    persistence_error = np.square(data["current"]-data["following"]).mean(1)
    result = {
        "complete": True, "post_hoc_exploratory": True, "validation_episodes": 75,
        "confirmation_opened": False, "additional_fitting": False,
        "frozen_readout_receipt_sha256": sha256(args.frozen_dir/"FROZEN.json"),
        "rmse_m": {"linear_world_then_known_geometry": float(np.sqrt(error.mean())),
                   "direct_goal_aligned_linear": float(np.sqrt(np.square(learned_linear-target).mean())),
                   "direct_goal_aligned_rbf": float(np.sqrt(np.square(learned_rbf-target).mean())),
                   "physical_persistence": float(np.sqrt(persistence_error.mean()))},
        "converted_linear_minus_direct_rbf_mse": per_episode_ci(error-np.square(learned_rbf-target).mean(1), groups),
        "world_linear_minus_persistence_mse": per_episode_ci(error-persistence_error, groups),
        "interpretation": "A frozen linear world readout plus known frame conversion is a competing explanation; no new nonlinear network mechanism is required by this readout result.",
        "anchor_caveat": "Uses actual current hand and goal to convert coordinates at evaluation; external physical anchors, not recovered model-native variables or an online controller.",
        "nonlinearity_not_ruled_out": True,
        "sources": data["sources"], "script_sha256": sha256(Path(__file__)),
    }
    write_json(args.output_dir/"coordinate_transform_null.json", result)
    np.savez_compressed(args.output_dir/"predictions.npz", converted_linear=converted, direct_rbf=learned_rbf,
                        target=target, seed=data["seed"], episode=data["episode"])
    write_json(args.output_dir/"DONE.json", {"complete": True, "outputs": [
        {"path": p.name, "sha256": sha256(p)} for p in sorted(args.output_dir.iterdir()) if p.is_file()]})
    print(json.dumps({"complete": True, "rmse_m": result["rmse_m"],
                      "paired_difference": result["converted_linear_minus_direct_rbf_mse"]}), flush=True)


if __name__ == "__main__":
    main()
