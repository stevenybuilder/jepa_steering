#!/usr/bin/env python3
"""Build the immutable 12-shard Reach-Wall trajectory manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from protocol import file_sha256, shard_for, split_for_episode, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("train-0010[2-4]-of-00126.parquet"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three verified Reach-Wall shards, found {paths}")

    rows = []
    files = []
    for path in paths:
        table = pq.read_table(path, columns=["task", "seed", "episode"])
        tasks = set(table.column("task").to_pylist())
        seeds = set(int(x) for x in table.column("seed").to_pylist())
        if tasks != {"mw-reach-wall"} or len(seeds) != 1 or table.num_rows != 100:
            raise RuntimeError(f"Unexpected metadata for {path}: tasks={tasks}, seeds={seeds}, rows={table.num_rows}")
        seed = seeds.pop()
        digest = file_sha256(path)
        files.append({"filename": path.name, "seed": seed, "rows": 100, "sha256": digest})
        for row_index, episode_value in enumerate(table.column("episode").to_pylist()):
            episode = int(episode_value)
            rows.append(
                {
                    "task": "mw-reach-wall",
                    "seed": seed,
                    "episode": episode,
                    "row": row_index,
                    "split": split_for_episode(episode),
                    "shard": shard_for(seed, episode),
                    "parquet": path.name,
                    "parquet_sha256": digest,
                }
            )
    rows.sort(key=lambda row: (row["seed"], row["episode"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    for shard in range(12):
        selected = [row for row in rows if row["shard"] == shard]
        (shard_dir / f"shard-{shard:02d}-of-12.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
        )
    counts = {name: sum(row["split"] == name for row in rows) for name in ("discovery", "validation", "confirmation")}
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": len(rows) == 300 and counts == {"discovery": 150, "validation": 75, "confirmation": 75},
        "task": "mw-reach-wall",
        "files": files,
        "trajectory_count": len(rows),
        "split_counts": counts,
        "shard_count": 12,
        "shard_rule": "(collection_seed - 1) * 4 + episode // 25",
        "split_rule": "episode 0:49 discovery, 50:74 validation, 75:99 confirmation within every seed",
        "manifest_sha256": file_sha256(manifest),
    }
    write_json_atomic(args.output_dir / "MANIFEST_DONE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["complete"]:
        raise SystemExit("Manifest integrity failure")


if __name__ == "__main__":
    main()
