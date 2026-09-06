#!/usr/bin/env python3
"""Run the unchanged native baseline collector on frozen new held IDs only.

This adapter adds identity/checkpoint checks and progress-only stdout. It does
not change initialization, native CEM, actions, episode length, or success logic.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from datetime import datetime, timezone


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            h.update(value)
    return h.hexdigest()


def validate_request(manifest, instance_id):
    allowed = manifest["allowed_new_episode_ids"]
    if allowed == list(range(24, 52)):
        shard_size = 7
    elif allowed == list(range(52, 62)):
        shard_size = 5
    else:
        raise RuntimeError("Only frozen24..51 expansion or separate52..61 tail may be collected")
    if manifest["seed_rules"]["environment_seed_base"] != 2026090500 or manifest["seed_rules"]["planner_seed_base"] != 90500:
        raise RuntimeError("Original seed rule changed")
    shards = manifest["shards"]
    all_ids = [episode for shard in shards for episode in shard["episode_ids"]]
    if sorted(all_ids) != allowed or len(set(all_ids)) != len(allowed):
        raise RuntimeError("Expansion shards overlap or miss an episode")
    selected = [shard for shard in shards if int(shard["instance_id"]) == instance_id]
    if len(selected) != 1 or len(selected[0]["episode_ids"]) != shard_size:
        raise RuntimeError("Unknown worker or incorrect shard size")
    return selected[0]


class ProgressOnly(io.TextIOBase):
    allowed = {"event", "episode", "completed", "replan", "environment_steps", "elapsed_seconds",
               "path", "sha256", "checkpoint_sha256", "instance_id", "episode_ids"}
    def __init__(self, target):
        self.target, self.buffer = target, ""
    def write(self, value):
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict) and "event" in row:
                self.target.write(json.dumps({k: v for k, v in row.items() if k in self.allowed}) + "\n")
                self.target.flush()
        return len(value)
    def flush(self):
        self.target.flush()


def write_exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    shard = validate_request(manifest, args.instance_id)
    config = args.repo / manifest["native_config"]["relative_path"]
    if file_hash(config) != manifest["native_config"]["sha256"]:
        raise RuntimeError("Native configuration hash changed")
    output = Path(shard["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    import collect_on_policy_bank as native
    original_loader = native.load_headless_metaworld
    def checked_loader(*positional, **kwargs):
        model, preprocessor, provenance = original_loader(*positional, **kwargs)
        actual = file_hash(Path(provenance["checkpoint"]))
        if actual != manifest["checkpoint"]["sha256"]:
            raise RuntimeError("Runtime checkpoint differs from frozen manifest")
        provenance["checkpoint_sha256"] = actual
        print(json.dumps({"event": "checkpoint_verified", "checkpoint_sha256": actual}), flush=True)
        return model, preprocessor, provenance
    native.load_headless_metaworld = checked_loader
    launch = {"created_utc": datetime.now(timezone.utc).isoformat(), "instance_id": args.instance_id,
              "episode_ids": shard["episode_ids"], "stage": "native unsteered held baseline generation",
              "manifest_sha256": file_hash(args.manifest), "adapter_sha256": file_hash(Path(__file__)),
              "native_collector_sha256": file_hash(Path(native.__file__)), "output": str(output),
              "environment_seeds": [2026090500+e for e in shard["episode_ids"]],
              "planner_seeds": [90500+e for e in shard["episode_ids"]]}
    write_exclusive(output / "LAUNCH.json", launch)
    print(json.dumps({"event": "expansion_started", "instance_id": args.instance_id, "episode_ids": shard["episode_ids"]}), flush=True)
    sys.argv = [str(Path(native.__file__)), "--repo", str(args.repo), "--config", str(config),
                "--output-dir", str(output), "--episodes", *map(str, shard["episode_ids"]),
                "--environment-seed-base", "2026090500", "--planner-seed-base", "90500"]
    with contextlib.redirect_stdout(ProgressOnly(sys.stdout)):
        native.main()
    # Receipt fields below exclude all held outcomes. No held tensor is loaded.
    done = json.loads((output / "DONE.json").read_text())
    if not done.get("complete") or sorted(done["episodes"]) != shard["episode_ids"]:
        raise RuntimeError("Incomplete native baseline receipt")
    entries = []
    for row in done["outputs"]:
        if row["environment_seed"] != 2026090500+row["episode"] or row["planner_seed"] != 90500+row["episode"]:
            raise RuntimeError("Native receipt seed mismatch")
        if file_hash(output / row["path"]) != row["sha256"]:
            raise RuntimeError("Native output hash mismatch")
        entries.append({key: row[key] for key in ("episode", "path", "sha256", "environment_seed", "planner_seed")})
    write_exclusive(output / "SEALED_DONE.json", {"complete": True, "scope": "new held baseline outputs; outcome fields excluded",
                                                 "native_done_sha256": file_hash(output / "DONE.json"), "outputs": entries})
    print(json.dumps({"event": "expansion_complete", "instance_id": args.instance_id,
                      "episode_ids": shard["episode_ids"], "sha256": file_hash(output / "SEALED_DONE.json")}), flush=True)


if __name__ == "__main__":
    main()
