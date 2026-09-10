#!/usr/bin/env python3
"""Locate the official MetaWorld parquet shards for one task.

This deliberately reads only task/seed/episode metadata.  Video decoding and model
inference belong to later stages, so a data-layout failure is cheap to diagnose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_shard(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    table = parquet.read_row_group(0, columns=["task", "seed", "episode"])
    tasks = sorted(set(table.column("task").to_pylist()))
    seeds = sorted(set(int(x) for x in table.column("seed").to_pylist()))
    episodes = table.column("episode").to_pylist()
    return {
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "tasks_first_row_group": tasks,
        "seeds_first_row_group": seeds,
        "episode_min_first_row_group": int(min(episodes)),
        "episode_max_first_row_group": int(max(episodes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="mw-reach-wall")
    parser.add_argument("--repo-id", default="facebook/jepa-wms")
    parser.add_argument("--prefix", default="metaworld/data/")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = sorted(
        name
        for name in list_repo_files(args.repo_id, repo_type="dataset")
        if name.startswith(args.prefix) and name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No parquet files under {args.prefix!r} in {args.repo_id}")

    def download_and_inspect(item: tuple[int, str]) -> dict[str, object]:
        index, filename = item
        local = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=args.cache_dir,
            )
        )
        metadata = inspect_shard(local)
        row: dict[str, object] = {"index": index, "filename": filename, **metadata}
        if args.task in metadata["tasks_first_row_group"]:
            row["local_path"] = str(local.resolve())
            row["sha256"] = sha256(local)
        return row

    scanned: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_and_inspect, item) for item in enumerate(files)]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            scanned.append(row)
            if args.task in row["tasks_first_row_group"]:
                print(json.dumps({"event": "match", **row}), flush=True)
            elif completed % 10 == 0:
                print(json.dumps({"event": "progress", "scanned": completed}), flush=True)

    scanned.sort(key=lambda row: int(row["index"]))
    matches = [row for row in scanned if args.task in row["tasks_first_row_group"]]

    seeds = sorted({seed for row in matches for seed in row["seeds_first_row_group"]})
    receipt = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "task": args.task,
        "candidate_file_count": len(files),
        "scanned_file_count": len(scanned),
        "matches": matches,
        "seeds": seeds,
        "complete": len(matches) == args.expected_shards and len(seeds) == args.expected_shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"event": "done", "output": str(args.output), "complete": receipt["complete"]}))
    if not receipt["complete"]:
        raise SystemExit(
            f"Expected {args.expected_shards} task shards/seeds, found {len(matches)} shards and {len(seeds)} seeds"
        )


if __name__ == "__main__":
    main()
