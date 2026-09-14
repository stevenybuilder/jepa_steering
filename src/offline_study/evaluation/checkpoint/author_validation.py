"""Metadata-only audit of the published validation population; never opens outcomes.

This is a preflight, not an evaluation runner or permission to consume a holdout.
Original manifests, exposure records, fit artifacts and results remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import torch
import yaml

from offline_study import VENDOR_COMMIT
from offline_study.core.protocol import sha256, write_json
from offline_study.models.vendor import use_vendor


CONFIGS = {
    "metaworld": "configs/vjepa_wm/mw_final_sweep/mw_4f_fsk5_ask1_r224_pred_AdaLN_ftprop_depth6_repro_2roll_save.yaml",
    "pusht": "configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml",
}
SOURCE_REVISION = "6116f042ae7ae4c8e3f1fd2f194f432615664182"
POOL_SIZES = {"metaworld": {"all": 12600}, "pusht": {"train": 18685, "val": 21}}


def official_partition(rows, kind, seed=234, train_fraction=.9):
    """Split the complete MW row order before filtering tasks; Push-T uses val/."""
    if kind not in CONFIGS or any(row["dataset"] != kind for row in rows):
        raise ValueError("Dataset mismatch")
    pools = {}
    for row in rows:
        pool = pools.setdefault(row["source_pool"], {})
        if row["index"] in pool:
            raise ValueError("Duplicate source row")
        pool[row["index"]] = row
    expected_pools = {"all"} if kind == "metaworld" else {"train", "val"}
    if set(pools) != expected_pools:
        raise ValueError("Missing or unexpected source pool")
    for pool in pools.values():
        if sorted(pool) != list(range(len(pool))):
            raise ValueError("Incomplete source row order; do not task-filter before splitting")
    if kind == "pusht":
        return ([pools["train"][i] for i in range(len(pools["train"]))],
                [pools["val"][i] for i in range(len(pools["val"]))])
    pool = pools["all"]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)).tolist()
    boundary = int(train_fraction * len(pool))
    return [pool[i] for i in order[:boundary]], [pool[i] for i in order[boundary:]]


def official_clips(rows, frames, stride, seed=234):
    """Mirror pinned TrajSlicerDataset including its conservative last start."""
    clips = [(row["trajectory_id"], start, start + frames * stride)
             for row in rows for start in range(row["length"] - frames * stride + 1)]
    order = torch.randperm(len(clips), generator=torch.Generator().manual_seed(seed)).tolist()
    return [clips[i] for i in order]


def require_disjoint_fit(fit_rows, validation_rows):
    overlap = {row["lineage_group"] for row in fit_rows} & {
        row["lineage_group"] for row in validation_rows}
    if overlap:
        raise ValueError(f"Fit overlaps {len(overlap)} author-validation lineage groups; refit required")


def approximate_standardized_mde(n, alpha=.05, power=.8):
    """Normal approximation for ONE paired contrast, in SDs of paired differences.

    Not observed power, not a multiplicity-adjusted calculation, and not a claim
    that a nominal number of lineage groups guarantees statistical independence.
    """
    if n < 2 or not 0 < alpha < 1 or not .5 < power < 1:
        raise ValueError("Invalid sensitivity assumptions")
    normal = NormalDist()
    return (normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) / math.sqrt(n)


def audit(rows, kind, config, registry=None, fits=None):
    if dict(Counter(row["source_pool"] for row in rows)) != POOL_SIZES[kind]:
        raise ValueError("Audit requires the complete released inventory, not a task subset")
    if {row.get("source_revision") for row in rows} != {SOURCE_REVISION}:
        raise ValueError("Unverified or mixed source revision")
    if len({row["trajectory_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate trajectory identity")
    if any(not row.get("lineage_group") or row["length"] < 1 for row in rows):
        raise ValueError("Missing lineage or invalid length")
    data = config["data"]
    custom, validation = data["custom"], data["validation"]
    train, val = official_partition(rows, kind, data["seed"], custom["split_ratio"])
    frames, stride = validation["num_frames_val"], custom["frameskip"]
    horizon = config["meta"]["data_traj_rollout_eval"]["data_traj_eval_rollout_steps"]
    if frames <= horizon:
        raise ValueError("Validation clip cannot supply the configured rollout")
    clips = official_clips(val, frames, stride, data["seed"])
    clips_per_id = Counter(clip[0] for clip in clips)
    val_groups = {row["lineage_group"] for row in val}
    train_groups = {row["lineage_group"] for row in train}
    entries = (registry or {}).get("trajectories", {})
    protected_groups = {row["lineage_group"] for row in rows
                        if row["split"] == "holdout" or
                        entries.get(row["trajectory_id"], {}).get("use") == "protected"}
    tasks = {}
    for task in sorted({row["task"] for row in val}):
        selected = [row for row in val if row["task"] == task]
        groups = {row["lineage_group"] for row in selected}
        task_clips = sum(clips_per_id[row["trajectory_id"]] for row in selected)
        tasks[task] = {
            "source_rows_available": sum(row["task"] == task for row in rows),
            "validation_rows": len(selected), "validation_lineage_groups": len(groups),
            "validation_clips": task_clips,
            "h6_prefix_rollouts_per_arm_full_pass": task_clips * (frames - horizon),
            "original_split_counts": dict(Counter(row["split"] for row in selected)),
            "protected_ids": [row["trajectory_id"] for row in selected
                              if row["lineage_group"] in protected_groups],
            "not_cleared_for_development_ids": [row["trajectory_id"] for row in selected
                if not (entries.get(row["trajectory_id"], {}).get("use") == "development"
                        and entries.get(row["trajectory_id"], {}).get("evidence"))],
            "official_train_shared_lineage_groups": sorted(groups & train_groups),
        }
    return {
        "status": "metadata_preflight_only_NOT_execution_authorization",
        "authors_evaluation_reproduced": False,
        "source_revision": SOURCE_REVISION, "vendor_commit": VENDOR_COMMIT,
        "source_rows": len(rows), "validation_rows": len(val),
        "validation_lineage_groups": len(val_groups), "validation_clips": len(clips),
        "validation_source_order": [row["trajectory_id"] for row in val],
        "validation_clip_order": clips,
        "prefixes_per_clip": frames - horizon,
        "validation_batch_size_per_rank": validation["val_dataset_batch_size"],
        "validation_drop_last": validation["val_dataset_drop_last"],
        "rollout_settings": config["meta"]["data_traj_rollout_eval"],
        "data_augmentation": config["data_aug"], "precision": config["meta"]["dtype"],
        "tasks": tasks,
        "fit_validation_overlap": {name: sorted({row["lineage_group"] for row in fit} & val_groups)
                                   for name, fit in (fits or {}).items()},
        "remaining_gates": ["freeze executable matched evaluator and unchanged intervention arms",
            "verify preprocessing, context, and metric parity against upstream",
            "refit without validation-family overlap under the matched context",
            "verify exposure authorization, including protected cohorts",
            "execute and verify fresh result receipts"],
        "limitations": ["Full-pass clip counts describe available coverage, not the paper's observed batch history",
            "Clips, rollout prefixes, arms and repeated checkpoints do not create independent families",
            "Existing study protection and exposure are not changed by the official partition",
            "Official MW row split itself shares some numerical lineages across train and validation",
            "Frozen released-checkpoint evaluation does not reproduce multi-seed training curves"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--fit-selection", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vendor = use_vendor(args.vendor)
    config_path = vendor / CONFIGS[args.dataset]
    config = yaml.safe_load(config_path.read_text())
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    registry = json.loads(args.registry.read_text()) if args.registry else None
    if registry and registry.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("Exposure registry does not bind this manifest")
    fits = {str(path): json.loads(path.read_text()) for path in args.fit_selection}
    report = audit(rows, args.dataset, config, registry, fits)
    sources = [args.manifest, config_path, *args.fit_selection]
    if args.registry:
        sources.append(args.registry)
    report["input_sha256"] = {str(path): sha256(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "preflight.json", report)
    print(json.dumps({key: report[key] for key in
                     ("status", "source_rows", "validation_rows", "validation_lineage_groups", "validation_clips")}))


if __name__ == "__main__":
    main()
