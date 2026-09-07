"""Read-only, outcome-blind cross-worker stimulus audit of running fixed panels."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from offline_study.protocol import sha256, write_json

ROOT = "/workspace/jepa-runtime/behavioral-development-20260907-v1"
KEY = "/tmp/jepa_vast_50123620_ed25519"
WORKERS = {50125440: ("209.146.116.50", 45353), 50195621: ("212.50.233.141", 42117)}
REMOTE = r'''
import hashlib,json
from pathlib import Path
root=Path("/workspace/jepa-runtime/behavioral-development-20260907-v1")
rows=[]
for path in sorted(p for p in root.glob("*/*/*/episode-*.json")
                   if p.parent.name.startswith(("shard-", "rank-"))):
    data=path.read_bytes(); r=json.loads(data)
    rows.append({"task":path.parts[-4], "arm":r["arm"], "episode":r["episode"],
        "logical_rank":r["logical_rank"], "environment_seed":r["environment_seed"],
        "initial_state_vector":r["initial_state_vector"],
        "initial_sha256":r["result"]["initial_sha256"], "goal_sha256":r["result"]["goal_sha256"],
        "episode_file":str(path), "episode_file_sha256":hashlib.sha256(data).hexdigest()})
print(json.dumps(rows))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.freeze / "FROZEN.json").read_text())
    if sha256(args.freeze / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Changed scientific freeze")
    protocol = json.loads((args.freeze / "protocol.json").read_text())
    schedule = {r["episode"]: r for r in protocol["episodes"]}
    indexed, counts = {}, {}
    for instance, (ip, port) in WORKERS.items():
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", KEY, "-p", str(port), "root@" + ip, "python3 -"],
            input=REMOTE, text=True, capture_output=True, check=True)
        rows = json.loads(result.stdout)
        for row in rows:
            key = row["task"], row["arm"], row["episode"]
            if key in indexed:
                raise ValueError("Duplicated episode assignment across workers")
            expected = schedule[row["episode"]]
            if any(row[k] != expected[k] for k in ("logical_rank", "environment_seed")):
                raise ValueError("Changed scenario identity")
            row["instance"] = instance
            indexed[key] = row
            counter = f"{instance}/{row['task']}/{row['arm']}"
            counts[counter] = counts.get(counter, 0) + 1
    pairs, missing, mismatches = [], [], []
    for (task, arm, episode), row in indexed.items():
        if arm == "native":
            continue
        baseline = indexed.get((task, "native", episode))
        if baseline is None:
            missing.append([task, arm, episode])
            continue
        fields = ("initial_state_vector", "initial_sha256", "goal_sha256", "logical_rank", "environment_seed")
        if any(row[k] != baseline[k] for k in fields):
            mismatches.append({"task": task, "arm": arm, "episode": episode,
                "fields": [k for k in fields if row[k] != baseline[k]],
                "baseline": baseline, "candidate": row})
            continue
        pairs.append({"task": task, "arm": arm, "episode": episode,
                      "baseline": baseline, "candidate": row, "stimuli_exact": True})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "pairs.json", pairs)
    write_json(args.output / "mismatches.json", mismatches)
    write_json(args.output / "report.json", {"status": "stimulus_pairing_failed" if mismatches else "partial_stimulus_pairing_verified_not_efficacy",
        "freeze_sha256": frozen["protocol_sha256"], "pairs_sha256": sha256(args.output / "pairs.json"),
        "completed_episode_counts": counts, "paired_candidate_episodes": len(pairs),
        "pending_baseline_partners": missing, "partial_snapshot": True,
        "mismatched_candidate_episodes": len(mismatches),
        "mismatches_sha256": sha256(args.output / "mismatches.json"),
        "candidate_success_values_read_locally": False, "outcome_selection_performed": False,
        "full_behavioral_panel_complete": False, "fresh_confirmation": False})
    write_json(args.output / ("FAILED.json" if mismatches else "DONE.json"),
               {"report_sha256": sha256(args.output / "report.json")})
    print(json.dumps({"paired_candidate_episodes": len(pairs), "counts": counts, "pending": len(missing),
                     "mismatches": [{k: r[k] for k in ("task", "arm", "episode", "fields")} for r in mismatches]}))
    if mismatches:
        raise ValueError("Unpaired goals recorded; no efficacy analysis authorized")


if __name__ == "__main__":
    main()
