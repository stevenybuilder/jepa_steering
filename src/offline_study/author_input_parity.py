"""Fit-only tensor parity against the actual upstream dataset slicer, without a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .author_runtime import encoded_batches, open_normalized_dataset, validate_cohort
from .protocol import sha256, write_json
from .vendor import use_vendor


class IdentityEncoder:
    def encode_clip(self, observations, actions):
        return {**observations, "action": actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vendor", "cohort", "data-root", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        use_vendor(args.vendor)
        from app.plan_common.datasets.traj_dset import TrajSlicerDataset, TrajSubset
        cohort = json.loads(args.cohort.read_text())
        validate_cohort(cohort)
        config = cohort["reference_config"]
        selected = cohort["fit"][:1]
        dataset = open_normalized_dataset(cohort["dataset"], args.data_root, config, True)
        sliced = TrajSlicerDataset(TrajSubset(dataset, [selected[0]["index"]]),
            config["data"]["validation"]["num_frames_val"], config["data"]["custom"]["frameskip"],
            config["data"]["custom"]["action_skip"], generator=torch.Generator().manual_seed(config["data"]["seed"]))
        slice_lookup = {start: i for i, (_, start, _) in enumerate(sliced.slices)}
        checked = []
        for metadata, batch in encoded_batches(IdentityEncoder(), dataset, selected, config, fitting=True):
            for i, group in enumerate(metadata):
                start = group[0]["clip_start"]
                obs, actions, _, _ = sliced[slice_lookup[start]]
                for name, expected in {**obs, "action": actions}.items():
                    if not torch.equal(batch[name][i], expected):
                        raise ValueError(f"Official input parity failed: {name}, clip {start}")
                checked.append(start)
        if not checked:
            raise ValueError("No input parity clips checked")
        write_json(args.output / "report.json", {"status": "real_fit_only_input_tensor_parity_passed",
            "trajectory_id": selected[0]["trajectory_id"], "clip_starts": checked,
            "compared": ["normalized_transformed_visual", "normalized_proprio", "normalized_concatenated_actions"],
            "bitwise_equal": True, "actual_upstream_slicer": True, "cohort_sha256": sha256(args.cohort),
            "development_or_protected_outcomes_accessed": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
