#!/usr/bin/env python3
"""Read-only sealed receipt/hash monitor for exactly four Reach + four Push."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from monitor_steering_preparation import execute, provider_state, remote_progress, save, sha, verify_local


def check(job, project, provider, board):
    result = verify_local(job, project)
    if result: return result
    line = next((r for r in board.read_text().splitlines() if r.startswith("| `" + str(job["instance_id"]) + "` |")), "")
    if "rep_geometry_transcoder" not in line or job["instance_id"] not in provider: raise RuntimeError("Lease check failed")
    status = remote_progress(job)
    if status["sealed_done"]:
        local = project / job["local_directory"]
        local.parent.mkdir(parents=True, exist_ok=True)
        execute(["scp", "-O", "-r", "-P", str(job["port"]), "root@" + job["host"] + ":" + job["remote_directory"], str(local.parent) + "/"], 600)
        result = verify_local(job, project)
        if result: return result
        raise RuntimeError("Copy did not produce verified receipt")
    if not status["active"]: raise RuntimeError("No native process without DONE; no automatic experiment retry")
    return {"complete": False, "task": job["task"], "instance_id": job["instance_id"], **status}


def main():
    p = argparse.ArgumentParser()
    for arg in ("manifest", "project", "board", "output"): p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.manifest.read_text())
    jobs = config["jobs"]
    if sorted(e for j in jobs if j["task"] == "reach_wall" for e in j["episode_ids"]) != list(range(62, 66)): raise ValueError("Wrong Reach IDs")
    if sorted(e for j in jobs if j["task"] == "pusht_scripted_v2" for e in j["episode_ids"]) != list(range(50, 54)): raise ValueError("Wrong Push IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        provider = provider_state()
        def safe(job):
            try: return check(job, args.project, provider, args.board)
            except Exception as exc: return {"complete": False, "instance_id": job["instance_id"], "task": job["task"], "error": str(exc)}
        with ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(safe, jobs))
        value = {"updated_utc": datetime.now(timezone.utc).isoformat(), "complete": all(r["complete"] for r in rows),
                 "rows": rows, "held_tensors_deserialized": False, "manifest_sha256": sha(args.manifest),
                 "script_sha256": sha(Path(__file__)), "scope": "eight extra unsteered native baselines only; no efficacy claim",
                 "verified_files": {task: sum(r.get("files", 0) for r in rows if r["task"] == task) for task in ("reach_wall", "pusht_scripted_v2")}}
        save(args.output / "PROGRESS.json", value)
        if value["complete"]:
            save(args.output / "DONE.json", value)
            return
        print(json.dumps({k: value[k] for k in ("updated_utc", "complete", "verified_files")}), flush=True)
        time.sleep(45)


if __name__ == "__main__": main()
