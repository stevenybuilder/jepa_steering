#!/usr/bin/env python3
"""Merge disjoint Panel-P capture shards into one fitter-ready capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_captures(sources: list[Path], out: Path, repo: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite merge output: {out}")
    if len(sources) < 2:
        raise ValueError("at least two capture shards are required")

    loaded = []
    for source in sources:
        index_path = source / "activations" / "index.json"
        if not (source / "DONE.json").is_file() or not index_path.is_file():
            raise ValueError(f"incomplete capture shard: {source}")
        index = json.loads(index_path.read_text())
        episodes = read_jsonl(source / "episodes.jsonl")
        by_ep = {int(row["ep"]): row for row in episodes}
        if len(by_ep) != len(episodes) or len(episodes) != len(index.get("episodes", [])):
            raise ValueError(f"episode/index cardinality mismatch: {source}")
        loaded.append((source.resolve(), index, by_ep))

    reference = loaded[0][1]
    identity_fields = ("config_sha256", "checkpoint_sha256", "repo_commit", "task_cfg", "model", "env")
    structural_fields = ("pre_outcome_window", "planner_capture_scope", "physical_pooling")
    for source, index, _episodes in loaded[1:]:
        for field in identity_fields:
            if index.get("meta", {}).get(field) != reference.get("meta", {}).get(field):
                raise ValueError(f"{source}: capture identity field {field!r} differs")
        for field in structural_fields:
            if index.get(field) != reference.get(field):
                raise ValueError(f"{source}: capture structure field {field!r} differs")

    out.mkdir(parents=True)
    (out / "activations").mkdir()
    (out / "behavior_traces").mkdir()
    merged_episodes: list[dict] = []
    merged_index_episodes: list[dict] = []
    merged_realizations: list[dict] = []
    seen_pairs: set[str] = set()

    for source, index, by_ep in loaded:
        realization_path = source / "realizations.jsonl"
        realizations = {
            int(row["ep"]): row for row in read_jsonl(realization_path)
        } if realization_path.is_file() else {}
        for activation_row in index["episodes"]:
            old_ep = int(activation_row["ep"])
            episode = dict(by_ep[old_ep])
            pair_id = str(episode.get("pair_id"))
            if not pair_id or pair_id == "None" or pair_id in seen_pairs:
                raise ValueError(f"missing or duplicate pair_id {pair_id!r}")
            seen_pairs.add(pair_id)
            new_ep = len(merged_episodes)

            activation_name = f"ep{new_ep:03d}.npz"
            source_activation = source / "activations" / activation_row["file"]
            shutil.copy2(source_activation, out / "activations" / activation_name)

            trace_rel = episode.get("behavior_trace")
            if not trace_rel:
                raise ValueError(f"{source}: episode {old_ep} has no behavior trace")
            trace_name = f"ep{new_ep:03d}.npz"
            source_trace = source / trace_rel
            destination_trace = out / "behavior_traces" / trace_name
            shutil.copy2(source_trace, destination_trace)

            episode.update({
                "ep": new_ep,
                "behavior_trace": f"behavior_traces/{trace_name}",
                "behavior_trace_sha256": sha256_file(destination_trace),
                "source_capture": str(source),
                "source_ep": old_ep,
            })
            merged_episodes.append(episode)
            merged_index_episodes.append({
                **activation_row,
                "ep": new_ep,
                "file": activation_name,
                "pair_id": pair_id,
                "source_capture": str(source),
                "source_ep": old_ep,
            })
            if old_ep in realizations:
                merged_realizations.append({
                    **realizations[old_ep],
                    "ep": new_ep,
                    "source_capture": str(source),
                    "source_ep": old_ep,
                })

    meta = dict(reference["meta"])
    meta.update({
        "label": "merged_capture",
        "task": meta.get("task_cfg"),
        "repo": str(repo.resolve()),
        "source_captures": [str(source) for source, _index, _episodes in loaded],
    })
    merged_index = {
        **reference,
        "meta": meta,
        "n_episodes": len(merged_index_episodes),
        "n_success": sum(int(row["success"]) for row in merged_index_episodes),
        "extra_forward_s": round(sum(float(index.get("extra_forward_s", 0.0)) for _source, index, _rows in loaded), 2),
        "capture_errors": [error for _source, index, _rows in loaded for error in index.get("capture_errors", [])],
        "episodes": merged_index_episodes,
    }
    (out / "activations" / "index.json").write_text(json.dumps(merged_index, indent=2) + "\n")
    (out / "episodes.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged_episodes))
    if merged_realizations:
        (out / "realizations.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged_realizations)
        )
    done = {
        "complete": True,
        "n_episodes": len(merged_episodes),
        "n_success": int(sum(int(row["success"]) for row in merged_episodes)),
        "n_failure": int(sum(not int(row["success"]) for row in merged_episodes)),
        "capture_index_sha256": sha256_file(out / "activations" / "index.json"),
        "source_capture_index_sha256": {
            str(source): sha256_file(source / "activations" / "index.json")
            for source, _index, _rows in loaded
        },
    }
    (out / "DONE.json").write_text(json.dumps(done, indent=2) + "\n")
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_captures(args.capture, args.out, args.repo), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
