#!/usr/bin/env python3
"""Create deterministic, provenance-bound shards of a frozen Panel-P manifest.

Rows are assigned round-robin by their source order.  ``pair_id`` and all stimulus
seeds/identities remain unchanged; only the evaluator-local ``ordinal`` is rewritten
to 0..len(shard)-1 so each derived manifest remains valid.  The source ordinal and
source-manifest hash are retained in every row and in a sidecar report.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

from public_panel_manifest import read_jsonl, rows_sha256, validate_rows, write_jsonl


def shard_rows(rows: list[dict], *, index: int, count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= index < count:
        raise ValueError(f"index must be in [0,{count - 1}]")
    source_hash = rows_sha256(rows)
    selected: list[dict] = []
    for source_index, source_row in enumerate(rows):
        if source_index % count != index:
            continue
        row = copy.deepcopy(source_row)
        row["source_ordinal"] = int(source_row["ordinal"])
        row["source_manifest_sha256"] = source_hash
        row["shard_index"] = int(index)
        row["shard_count"] = int(count)
        row["ordinal"] = len(selected)
        selected.append(row)
    if not selected:
        raise ValueError(f"shard {index}/{count} is empty")
    verdict = validate_rows(((f"shard-{index:02d}-of-{count:02d}", selected),))
    if not verdict["passed"]:
        raise ValueError(f"derived shard failed validation: {verdict['errors']}")
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, required=True)
    args = ap.parse_args()

    source = args.source.resolve()
    rows = read_jsonl(source)
    source_verdict = validate_rows(((source.stem, rows),))
    if not source_verdict["passed"]:
        raise SystemExit(f"invalid source manifest: {source_verdict['errors']}")
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty shard directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    all_pairs: set[str] = set()
    for index in range(args.count):
        shard = shard_rows(rows, index=index, count=args.count)
        path = out_dir / f"shard-{index:02d}-of-{args.count:02d}.jsonl"
        write_jsonl(path, shard)
        pair_ids = [str(row["pair_id"]) for row in shard]
        overlap = all_pairs.intersection(pair_ids)
        if overlap:
            raise RuntimeError(f"pair IDs assigned to multiple shards: {sorted(overlap)}")
        all_pairs.update(pair_ids)
        reports.append(
            {
                "index": index,
                "count": len(shard),
                "path": path.name,
                "sha256": rows_sha256(shard),
                "pair_ids": pair_ids,
                "source_ordinals": [int(row["source_ordinal"]) for row in shard],
            }
        )
    source_pairs = {str(row["pair_id"]) for row in rows}
    if all_pairs != source_pairs:
        raise RuntimeError(
            f"shard union mismatch: missing={sorted(source_pairs - all_pairs)}, "
            f"extra={sorted(all_pairs - source_pairs)}"
        )
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": rows_sha256(rows),
        "source_count": len(rows),
        "shard_count": args.count,
        "assignment": "source_index_mod_shard_count",
        "pair_ids_preserved": True,
        "shards": reports,
    }
    (out_dir / "SHARDS.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
