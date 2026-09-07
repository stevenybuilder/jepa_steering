"""Inventory real trajectories without fitting or evaluating any intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .protocol import sha256, study_split, validate_manifest, window_starts, write_json
from .vendor import open_dataset, use_vendor
from . import VENDOR_COMMIT


def _initial_state_group(value) -> str:
    """Stable Push-T family ID from the exact released initial state."""
    array = value.detach().cpu().contiguous().numpy()
    return f"pusht:initial-state:{hashlib.sha256(array.tobytes()).hexdigest()}"


def assign_study_splits(
    rows: list[dict], seed: int, *, holdout_eligible: bool = True
) -> dict[str, str]:
    """Split lineage groups; optionally return an exposed holdout to development."""
    eligible_groups = sorted({
        row["lineage_group"] for row in rows if row["source_pool"] != "val"
    })
    structural_membership = study_split(eligible_groups, seed)
    membership = {
        group: (split if holdout_eligible or split != "holdout" else "development")
        for group, split in structural_membership.items()
    }
    for row in rows:
        row["split"] = (
            membership[row["lineage_group"]]
            if row["source_pool"] != "val"
            else "external_reserve"
        )
    return membership


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--dataset", choices=["metaworld", "pusht"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-revision", required=True, help="Downloaded source revision, recorded verbatim")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seed", type=int, default=234)
    args = p.parse_args()
    if args.windows < 1:
        p.error("--windows must be positive")
    use_vendor(args.vendor)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        pools = ["all"] if args.dataset == "metaworld" else ["train", "val"]
        for pool in pools:
            dset = open_dataset(args.dataset, args.data_root, pool)
            # Reading a non-video column avoids decoding media just to inventory task names.
            tasks = list(dset.dataset["task"]) if args.dataset == "metaworld" else ["pusht"] * len(dset)
            lineage_groups = (
                [_initial_state_group(dset.states[i, 0]) for i in range(len(dset))]
                if args.dataset == "pusht"
                else [f"metaworld:trajectory:{i}" for i in range(len(dset))]
            )
            for i in range(len(dset)):
                length = int(dset.get_seq_length(i))
                rows.append(dict(trajectory_id=f"{args.dataset}:{pool}:{i}", dataset=args.dataset,
                                 source_pool=pool, index=i, task=tasks[i], length=length,
                                 lineage_group=lineage_groups[i],
                                 horizon=6, stride=5, windows_requested=args.windows,
                                 starts=window_starts(length, count=args.windows),
                                 exposure_status="unverified", world_model_exposure="unknown",
                                 source_revision=args.data_revision))
            del dset
        # The released Push-T train pool is already exposed across every lineage
        # family in this study. Preserve fit/development separation, but do not
        # manufacture a confirmation holdout by relabelling exposed families.
        holdout_eligible = args.dataset != "pusht"
        membership = assign_study_splits(
            rows, args.seed, holdout_eligible=holdout_eligible
        )
        structural_membership = study_split(sorted(membership), args.seed)
        validate_manifest(rows)
        manifest = args.output / "trajectories.jsonl"
        manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        limitations = [
            "Source revision and row order must be preserved; raw-media checksums not yet verified",
            "A split label alone does not establish that a cohort is historically untouched",
        ]
        if args.dataset == "pusht":
            limitations.extend([
                "Push-T supplied val is external_reserve",
                "Released Push-T holdout-labelled families are development-exposed and reassigned to development",
                "Push-T rollout rows sharing an exact initial state are repeated observations within one family",
            ])
        write_json(args.output / "inventory.json", dict(
            status="metadata_inventory_complete", dataset=args.dataset, data_root=str(args.data_root.resolve()),
            data_revision=args.data_revision, vendor_commit=VENDOR_COMMIT, source_seed=args.seed,
            split_policy=(
                "study_hash_by_lineage_group; Push-T released holdout groups returned to development"
                if args.dataset == "pusht" else
                "study_hash_90_10_with_inner_development_by_lineage_group; not exact authors' split"
            ),
            manifest_sha256=sha256(manifest), trajectories=len(rows),
            split_counts=dict(Counter(r["split"] for r in rows)),
            lineage_group_counts=dict(Counter(membership.values())),
            structural_lineage_group_counts=dict(Counter(structural_membership.values())),
            lineage_group_size_counts=dict(Counter(
                Counter(r["lineage_group"] for r in rows if r["source_pool"] != "val").values()
            )),
            task_counts=dict(Counter(r["task"] for r in rows)),
            eligible_window_counts=dict(Counter(r["split"] for r in rows for _ in r["starts"])),
            too_short_ids=[r["trajectory_id"] for r in rows if not r["starts"]],
            paper_total_targets={"metaworld": 12600, "pusht": 18500},
            source_trajectory_contents_hashed=False,
            lineage_policy=("exact_initial_state_sha256_family" if args.dataset == "pusht"
                            else "whole_trajectory"),
            confirmation_eligible_from_this_manifest=holdout_eligible,
            limitations=limitations,
        ))
        print(json.dumps({"status": "metadata_inventory_complete", "output": str(args.output), "trajectories": len(rows)}))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
