"""Audit released navigation splits and an exact selected-frame training loader.

CPU-only, no learned-model outcomes. Preserve native split and clip ordering,
action concatenation and transform RNG while avoiding unnecessary frame transforms.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .droid_native import assert_same, array_hash, verified_report
from .protocol import sha256, write_json
from .vendor import use_vendor


CONFIGS = {task: f"configs/vjepa_wm/{prefix}_sweep/{prefix}_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"
           for task, prefix in (("wall", "wall"), ("pointmaze", "mz"))}


class SelectedFrameSlicer:
    def __init__(self, original):
        from app.plan_common.datasets.traj_dset import TrajSlicerDataset, TrajSubset
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.transforms import VideoTransform
        if type(original) is not TrajSlicerDataset:
            raise ValueError("Require exact native slicer")
        base = original.dataset
        while type(base) is TrajSubset:
            base = base.dataset
        if type(base) not in (WallDataset, PointMazeDataset):
            raise ValueError("Only audited native navigation datasets supported")
        transform = base.transform
        if (type(transform) is not VideoTransform or transform.auto_augment or transform.motion_shift or
                transform.random_horizontal_flip or transform.reprob != 0 or transform.hwc or
                transform.do_255_to_1 or tuple(transform.random_resize_aspect_ratio) != (1., 1.) or
                tuple(transform.random_resize_scale) != (1.777, 1.777)):
            raise ValueError("Unvalidated stochastic/preprocessing configuration")
        if (original.num_frames != 4 or original.frameskip != 5 or original.action_skip != 1 or
                original.process_actions != "concat"):
            raise ValueError("This optimization is fixed to the released four-frame training clips")
        self.original, self.base = original, base
        self.is_maze = type(base) is PointMazeDataset

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        from app.plan_common.datasets.traj_dset import TrajSubset
        row, start, end = self.original.slices[index]
        subset = self.original.dataset
        while type(subset) is TrajSubset:
            row = int(subset.indices[row])
            subset = subset.dataset
        if subset is not self.base:
            raise ValueError("Dataset mapping changed")
        frames = list(range(start, end, 5))
        # mmap reads only selected pages of the uncompressed torch archive. Raw
        # input is read-only; division and transforms produce new tensors.
        image = torch.load(self.base.data_path / f"obses/episode_{row:03d}.pth",
                           map_location="cpu", weights_only=True, mmap=True)
        image = image[frames] / 255.0
        if self.is_maze:
            image = image.permute(0, 3, 1, 2)
        visual = self.base.transform(image)
        obs = {"visual": visual, "proprio": self.base.proprios[row, frames]}
        actions = self.base.actions[row, start:end].reshape(4, -1)
        state = self.base.states[row, frames]
        return obs, actions, state, torch.zeros(4, dtype=torch.float32)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    use_vendor(args.vendor)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        staged, staged_hash = verified_report(args.assets)
        if staged["status"] != "official_navigation_assets_staged_and_verified":
            raise ValueError("Unverified navigation source assets")
        write_json(args.output / "protocol.json", {"role": "cpu_input_metadata_and_exact_preprocessing_engineering",
            "source_sha256": sha256(Path(__file__)), "assets_report_sha256": staged_hash,
            "native_configs_sha256": {task: sha256(args.vendor / path) for task, path in CONFIGS.items()},
            "selection": "first eight native shuffled training clips, before model outcomes",
            "timing_repetitions": 2, "parity": "exact pixels/actions/states/rewards and RNG",
            "mmap_selected_frames": True, "model_evaluations": 0, "confirmation_authorized": False})
        from app.plan_common.datasets.transforms import make_transforms
        from app.plan_common.datasets.wall_dset import WallDataset
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.traj_dset import get_train_val_sliced
        rows = {}
        for task in ("wall", "pointmaze"):
            cfg = yaml.safe_load((args.vendor / CONFIGS[task]).read_text())
            data = cfg["data"]
            custom = data["custom"]
            if (custom["split_ratio"] != .9 or data["seed"] != 234 or data["img_size"] != 224 or
                    custom["num_hist"] + custom["num_pred"] != 4 or custom["normalize_action"] is not True):
                raise ValueError("Native navigation data protocol changed")
            raw_root = args.assets / "extracted" / task / ("wall_single" if task == "wall" else "point_maze")
            cls = WallDataset if task == "wall" else PointMazeDataset
            dset = cls(data_path=str(raw_root), n_rollout=None, normalize_action=True,
                       transform=make_transforms(img_size=224, **cfg["data_aug"]))
            train, validation, train_clips, val_clips = get_train_val_sliced(dset,
                train_fraction=.9, random_seed=234, num_frames=4,
                num_frames_val=data["validation"]["num_frames_val"], frameskip=5, action_skip=1)
            if set(train.indices) & set(validation.indices) or len(train) + len(validation) != len(dset):
                raise ValueError("Native trajectory partition is incomplete/overlapping")
            # Metadata-only lineage audit; never claim unique rows == independent
            # histories solely from absence of these exact duplicates.
            families = {}
            state_fingerprints = []
            for i in range(len(dset)):
                initial = dset.states[i, 0].reshape(-1)
                if task == "wall":
                    initial = torch.cat([initial, dset.door_locations[i, 0].reshape(-1),
                                         dset.wall_locations[i, 0].reshape(-1)])
                key = array_hash(initial.numpy())
                families.setdefault(key, []).append(i)
                state_fingerprints.append(array_hash(dset.states[i, :int(dset.get_seq_length(i))].numpy()))
            cross = [indices for indices in families.values()
                     if set(indices) & set(train.indices) and set(indices) & set(validation.indices)]
            optimized = SelectedFrameSlicer(train_clips)
            random.seed(2026090724)
            np.random.seed(2026090724)
            torch.manual_seed(2026090724)
            checks = []
            for repetition in range(2):
                for index in range(8):
                    before_rng = rng_state()
                    values, seconds, final_rngs = {}, {}, {}
                    order = ("native", "selected") if repetition == 0 else ("selected", "native")
                    for kind in order:
                        restore_rng(before_rng)
                        begin = time.monotonic()
                        values[kind] = (train_clips if kind == "native" else optimized)[index]
                        seconds[kind] = time.monotonic() - begin
                        final_rngs[kind] = rng_state()
                    assert_same(values["native"], values["selected"])
                    assert_same(final_rngs["native"], final_rngs["selected"])
                    checks.append({"clip_index": index, "repetition": repetition, "order": order,
                                   "seconds": seconds, "exact_values_and_rng": True})
            rows[task] = {"raw_trajectories": len(dset), "native_train_indices": list(train.indices),
                "native_validation_indices": list(validation.indices), "train_clips": len(train_clips),
                "validation_clips": len(val_clips), "state_shape": list(dset.states.shape),
                "action_shape": list(dset.actions.shape), "exact_initial_fingerprint_groups": len(families),
                "exact_state_trajectory_fingerprints": len(set(state_fingerprints)),
                "cross_split_initial_fingerprint_groups": cross,
                "independence_not_proven_by_fingerprint_uniqueness": True,
                "normalization_scope": "full released dataset before native split, as upstream",
                "preprocessing_checks": checks, "gpu_model_work": 0,
                "optimization_speedup_measured": sum(r["seconds"]["native"] for r in checks) /
                                                sum(r["seconds"]["selected"] for r in checks)}
            write_json(args.output / f"{task}.json", rows[task])
            print(json.dumps({"task": task, "rows": len(dset), "training": len(train),
                "validation": len(validation), "loader_speedup": rows[task]["optimization_speedup_measured"]}), flush=True)
        write_json(args.output / "report.json", {"status": "navigation_metadata_and_selected_frame_parity_passed",
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "task_reports_sha256": {task: sha256(args.output / f"{task}.json") for task in rows},
            "seconds": time.monotonic() - started, "model_evaluations": 0,
            "training_ran": False, "confirmation_authorized": False})
        write_json(args.output / "DONE.json", {"report_sha256": sha256(args.output / "report.json")})
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"error": str(exc), "model_evaluations": 0})
        raise


if __name__ == "__main__":
    main()
