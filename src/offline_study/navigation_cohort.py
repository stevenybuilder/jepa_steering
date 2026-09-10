"""Author-split navigation cohorts for unchanged offline fits and comparisons.

New tasks only: never relabel any pre-existing MetaWorld/Push-T exposure registry.
Keep all released validation trajectories, exclude their initial-state families
from fitting, and bind every accessed video plus metadata before model outcomes.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from .author_runtime import examples, open_normalized_dataset, validate_cohort
from .droid_native import array_hash, verified_report
from .navigation_input_check import CONFIGS
from .operator_fit import FIT_SEED, select_fit_rows
from .protocol import sha256, write_json
from .vendor import use_vendor


COUNTS = {"wall": (1920, 1728, 192, 2112), "pointmaze": (2000, 1800, 200, 12200)}


def prepare(vendor, assets, input_check, task, output):
    use_vendor(vendor)
    from app.plan_common.datasets.traj_dset import get_train_val_sliced
    staged, staged_hash = verified_report(assets)
    audited, audited_hash = verified_report(input_check)
    ip = json.loads((input_check / "protocol.json").read_text())
    proof = json.loads((input_check / f"{task}.json").read_text())
    if (staged["status"] != "official_navigation_assets_staged_and_verified" or
            audited["status"] != "navigation_metadata_and_selected_frame_parity_passed" or
            audited["protocol_sha256"] != sha256(input_check / "protocol.json") or
            audited["task_reports_sha256"][task] != sha256(input_check / f"{task}.json") or
            ip["assets_report_sha256"] != staged_hash or
            ip["native_configs_sha256"][task] != sha256(vendor / CONFIGS[task]) or
            proof["cross_split_initial_fingerprint_groups"]):
        raise ValueError("Input provenance changed or native train/validation families overlap")
    cfg = yaml.safe_load((vendor / CONFIGS[task]).read_text())
    root = assets / "extracted" / task / ("wall_single" if task == "wall" else "point_maze")
    dset = open_normalized_dataset(task, root, cfg, True)
    train, val, _, val_clips = get_train_val_sliced(dset, train_fraction=.9,
        random_seed=234, num_frames=4, num_frames_val=8, frameskip=5, action_skip=1)
    if ((len(dset), len(train), len(val), len(val_clips)) != COUNTS[task] or
            list(train.indices) != proof["native_train_indices"] or
            list(val.indices) != proof["native_validation_indices"]):
        raise ValueError("Native navigation split, order or count differs")
    train_ids = set(train.indices)
    rows = []
    for index in range(len(dset)):
        initial = dset.states[index, 0].reshape(-1)
        if task == "wall":
            initial = torch.cat([initial, dset.door_locations[index, 0].reshape(-1),
                                 dset.wall_locations[index, 0].reshape(-1)])
        rows.append({"index": index, "trajectory_id": f"{task}/released/{index:05d}", "task": task,
            "lineage_group": task + "/initial/" + array_hash(initial.numpy()),
            "length": int(dset.get_seq_length(index)), "split": "fit" if index in train_ids else "development",
            "source_pool": "released_native_train" if index in train_ids else "released_native_validation"})
    evaluation = [rows[int(index)] for index in val.indices]
    all_val = {row["lineage_group"] for row in evaluation}
    eligible = [rows[int(index)] for index in train.indices if rows[int(index)]["lineage_group"] not in all_val]
    fit = select_fit_rows(eligible, task, 128, FIT_SEED)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "source_manifest.json", rows)
    files = {path.name: sha256(path) for path in sorted(root.glob("*.pth"))}
    for row in fit + evaluation:
        name = f"obses/episode_{row['index']:03d}.pth"
        files[name] = sha256(root / name)
    write_json(output / "input_files.json", files)
    cohort = {"schema_version": 1, "task": task, "dataset": task,
        "source_manifest_sha256": sha256(output / "source_manifest.json"),
        "source_input_files_sha256": sha256(output / "input_files.json"),
        "source_assets_report_sha256": staged_hash, "input_check_report_sha256": audited_hash,
        "fit": fit, "evaluation": evaluation, "protected_official_validation": [],
        "all_author_validation_groups": sorted(all_val), "all_protected_groups": [],
        "coverage": {"evaluated_rows": len(evaluation), "official_rows": len(evaluation), "complete_author_split": True},
        "reference_config": cfg, "upstream_config_sha256": sha256(vendor / CONFIGS[task]),
        "role": "new_task_full_author_validation_development_replication",
        "evaluation_permission": "new_navigation_tasks_user_authorized_development_only",
        "fresh_confirmation": False,
        "access_review": "New task data; source train/validation split retained. Base-1 simulator confirmation remains separate and unopened. No legacy protected cohort rewritten.",
        "fit_sampling": "unchanged128 hash-selected initial-state families, four clip/prefix examples each",
        "evaluation_sampling": "all official eight-frame validation clips and both H6 prefixes",
        "primary_precision": "bfloat16", "secondary_precision": "float32",
        "precision_selection_forbidden": True, "initial_fingerprint_independence_proof": False,
        "model_training_exposure_distinct_from_intervention_fit": True}
    validate_cohort(cohort)
    if len(examples(evaluation, cfg)) != 2 * COUNTS[task][3]:
        raise ValueError("Incorrect complete offline prefix count")
    write_json(output / "cohort.json", cohort)
    write_json(output / "FROZEN.json", {"cohort_sha256": sha256(output / "cohort.json"),
        "input_files_sha256": sha256(output / "input_files.json"), "model_runs": 0,
        "confirmation_authorized": False})
    print(json.dumps({"task": task, "fit_families": len(fit), "validation_rows": len(evaluation),
        "validation_clips": COUNTS[task][3], "H6_prefixes": 2 * COUNTS[task][3]}), flush=True)


def verify_inputs(cohort_path, root):
    cohort = json.loads(cohort_path.read_text())
    directory = cohort_path.parent
    frozen = json.loads((directory / "FROZEN.json").read_text())
    if (frozen["cohort_sha256"] != sha256(cohort_path) or
            cohort["source_input_files_sha256"] != sha256(directory / "input_files.json") or
            cohort["source_manifest_sha256"] != sha256(directory / "source_manifest.json")):
        raise ValueError("Navigation cohort/input freeze changed")
    for name, digest in json.loads((directory / "input_files.json").read_text()).items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha256(root / path) != digest:
            raise ValueError("Navigation raw input changed: " + name)
    validate_cohort(cohort)
    return cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "assets", "input-check", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", choices=tuple(COUNTS), required=True)
    args = parser.parse_args()
    prepare(args.vendor, args.assets, args.input_check, args.task, args.output)


if __name__ == "__main__":
    main()
