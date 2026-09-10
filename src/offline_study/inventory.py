"""Inventory real trajectories without fitting or evaluating any intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .protocol import sha256, study_split, validate_manifest, window_starts, write_json
from .vendor import open_dataset, use_vendor
from . import VENDOR_COMMIT


def _initial_state_group(value) -> str:
    """Stable Push-T family ID from the exact released initial state."""
    array = value.detach().cpu().contiguous().numpy()
    return f"pusht:initial-state:{hashlib.sha256(array.tobytes()).hexdigest()}"


def _update_tensor_digest(digest, label: str, value) -> None:
    """Hash tensor contents without relying on Python object serialization."""
    array = value.detach().cpu().contiguous().numpy()
    digest.update(label.encode())
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())


def _metaworld_trajectory_group(task: str, states, actions) -> str:
    """Stable lineage ID for an exact numerical state/action trajectory."""
    digest = hashlib.sha256()
    digest.update(task.encode())
    _update_tensor_digest(digest, "states", states)
    _update_tensor_digest(digest, "actions", actions)
    return f"metaworld:state-action:{digest.hexdigest()}"


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


def assign_metaworld_splits(
    rows: list[dict], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Preserve the legacy row split while coalescing exact duplicate groups.

    The first MetaWorld baseline split one released row at a time. Re-hashing new
    content-derived group IDs would reshuffle every row and could relabel exposed
    development rows as holdout. Instead, reconstruct that legacy split, then give
    each duplicate group one conservative assignment: development wins because it
    has been measured; otherwise an untouched holdout wins over unused fit.
    """
    legacy_units = sorted({row["legacy_split_unit"] for row in rows})
    legacy_membership = study_split(legacy_units, seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["lineage_group"]].append(
            legacy_membership[row["legacy_split_unit"]]
        )
    precedence = {"fit": 0, "holdout": 1, "development": 2}
    membership = {
        group: max(splits, key=precedence.__getitem__)
        for group, splits in grouped.items()
    }
    for row in rows:
        row["split"] = membership[row["lineage_group"]]
    return membership, legacy_membership


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
            source_seeds = list(dset.dataset["seed"]) if args.dataset == "metaworld" else [None] * len(dset)
            source_episodes = list(dset.dataset["episode"]) if args.dataset == "metaworld" else [None] * len(dset)
            lineage_groups = (
                [_initial_state_group(dset.states[i, 0]) for i in range(len(dset))]
                if args.dataset == "pusht"
                else [
                    _metaworld_trajectory_group(
                        tasks[i], dset.states[i], dset.actions[i]
                    )
                    for i in range(len(dset))
                ]
            )
            for i in range(len(dset)):
                length = int(dset.get_seq_length(i))
                rows.append(dict(trajectory_id=f"{args.dataset}:{pool}:{i}", dataset=args.dataset,
                                 source_pool=pool, index=i, task=tasks[i], length=length,
                                 lineage_group=lineage_groups[i],
                                 legacy_split_unit=(f"metaworld:{pool}:{i}"
                                                    if args.dataset == "metaworld" else None),
                                 source_seed=source_seeds[i], source_episode=source_episodes[i],
                                 horizon=6, stride=5, windows_requested=args.windows,
                                 starts=window_starts(length, count=args.windows),
                                 exposure_status="unverified", world_model_exposure="unknown",
                                 source_revision=args.data_revision))
            del dset
        # The released Push-T train pool is already exposed across every lineage
        # family in this study. Preserve fit/development separation, but do not
        # manufacture a confirmation holdout by relabelling exposed families.
        holdout_eligible = args.dataset != "pusht"
        if args.dataset == "metaworld":
            membership, legacy_membership = assign_metaworld_splits(rows, args.seed)
            structural_membership = membership
        else:
            membership = assign_study_splits(
                rows, args.seed, holdout_eligible=holdout_eligible
            )
            structural_membership = study_split(sorted(membership), args.seed)
            legacy_membership = None
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
        else:
            limitations.extend([
                "MetaWorld exact numerical state/action duplicates share one lineage group",
                "The corrected group split preserves historical development exposure instead of re-hashing all rows",
                "Video bytes are not part of the MetaWorld lineage digest",
            ])
        group_sizes = Counter(
            r["lineage_group"] for r in rows if r["source_pool"] != "val"
        )
        write_json(args.output / "inventory.json", dict(
            status="metadata_inventory_complete", dataset=args.dataset, data_root=str(args.data_root.resolve()),
            data_revision=args.data_revision, vendor_commit=VENDOR_COMMIT, source_seed=args.seed,
            split_policy=(
                "study_hash_by_lineage_group; Push-T released holdout groups returned to development"
                if args.dataset == "pusht" else
                "legacy_study_hash_by_row_then_exposure_preserving_exact_duplicate_group_coalescing; not exact authors' split"
            ),
            manifest_sha256=sha256(manifest), trajectories=len(rows),
            split_counts=dict(Counter(r["split"] for r in rows)),
            lineage_group_counts=dict(Counter(membership.values())),
            structural_lineage_group_counts=dict(Counter(structural_membership.values())),
            lineage_group_size_counts=dict(Counter(
                group_sizes.values()
            )),
            duplicated_lineage_groups=sum(size > 1 for size in group_sizes.values()),
            duplicate_rows=sum(size - 1 for size in group_sizes.values()),
            legacy_row_split_counts=(
                dict(Counter(legacy_membership.values()))
                if legacy_membership is not None else None
            ),
            task_counts=dict(Counter(r["task"] for r in rows)),
            eligible_window_counts=dict(Counter(r["split"] for r in rows for _ in r["starts"])),
            too_short_ids=[r["trajectory_id"] for r in rows if not r["starts"]],
            paper_total_targets={"metaworld": 12600, "pusht": 18500},
            source_trajectory_contents_hashed=args.dataset == "metaworld",
            lineage_policy=("exact_initial_state_sha256_family" if args.dataset == "pusht"
                            else "exact_task_state_action_tensor_sha256"),
            confirmation_eligible_from_this_manifest=holdout_eligible,
            limitations=limitations,
        ))
        print(json.dumps({"status": "metadata_inventory_complete", "output": str(args.output), "trajectories": len(rows)}))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
