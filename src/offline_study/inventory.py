"""Inventory real trajectories without fitting or evaluating any intervention."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .protocol import sha256, study_split, validate_manifest, window_starts, write_json
from .vendor import open_dataset, use_vendor
from . import VENDOR_COMMIT


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
            for i in range(len(dset)):
                length = int(dset.get_seq_length(i))
                rows.append(dict(trajectory_id=f"{args.dataset}:{pool}:{i}", dataset=args.dataset,
                                 source_pool=pool, index=i, task=tasks[i], length=length,
                                 horizon=6, stride=5, windows_requested=args.windows,
                                 starts=window_starts(length, count=args.windows),
                                 exposure_status="unverified", world_model_exposure="unknown",
                                 source_revision=args.data_revision))
            del dset
        eligible = [r["trajectory_id"] for r in rows if r["source_pool"] != "val"]
        membership = study_split(eligible, args.seed)
        for row in rows:
            row["split"] = membership.get(row["trajectory_id"], "external_reserve")
        validate_manifest(rows)
        manifest = args.output / "trajectories.jsonl"
        manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        write_json(args.output / "inventory.json", dict(
            status="metadata_inventory_complete", dataset=args.dataset, data_root=str(args.data_root.resolve()),
            data_revision=args.data_revision, vendor_commit=VENDOR_COMMIT, source_seed=args.seed,
            split_policy="study_hash_90_10_with_inner_development; not exact authors' split",
            manifest_sha256=sha256(manifest), trajectories=len(rows),
            split_counts=dict(Counter(r["split"] for r in rows)),
            task_counts=dict(Counter(r["task"] for r in rows)),
            eligible_window_counts=dict(Counter(r["split"] for r in rows for _ in r["starts"])),
            too_short_ids=[r["trajectory_id"] for r in rows if not r["starts"]],
            paper_total_targets={"metaworld": 12600, "pusht": 18500},
            source_trajectory_contents_hashed=False,
            limitations=["Source revision and row order must be preserved; raw-media checksums not yet verified",
                         "No claim that provisional holdout is historically unseen",
                         "Push-T supplied val is external_reserve; 90/10 is applied to supplied train only"]
        ))
        print(json.dumps({"status": "metadata_inventory_complete", "output": str(args.output), "trajectories": len(rows)}))
    except Exception as exc:
        write_json(args.output / "FAILED.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
