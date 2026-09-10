"""Bounded common-action diagnostic, not full CEM or confirmation.

CPU analysis is deliberately independent of the GPU/simulator dependencies.
See docs/PLANNER_DECISION_DIAGNOSTIC.md for the prospectively fixed estimand.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ("reach", "reach-wall")
ARMS = ("native", "fixed_rank4", "matched_random_fixed_rank4",
        "coupling_only", "matched_random_coupling")
CONTRASTS = (("fixed_rank4", "native"), ("fixed_rank4", "matched_random_fixed_rank4"),
             ("coupling_only", "native"), ("coupling_only", "matched_random_coupling"))
ROLE = "common_initial_candidate_decision_development_v1"
ANALYSIS = {"replicates": 20000, "seed": 2026091022, "family": 8,
            "endpoint": "terminal_ee_distance", "confidence": .95,
            "unit": "whole_scenario", "confirmation": False}


def candidate_seed(task, environment_seed):
    return int(hashlib.sha256(f"jepa-decision-v1/{task}/{environment_seed}".encode()).hexdigest()[:8], 16)


def score_summary(scores):
    a = np.asarray(scores, dtype=np.float64)
    if a.shape != (300,) or not np.isfinite(a).all():
        raise ValueError("Exactly 300 finite candidate scores required")
    order = np.argsort(a, kind="stable")
    return {"selected": int(order[0]), "margin": float(a[order[1]] - a[order[0]]),
            "top10": order[:10].tolist()}


def ranks(a):
    a = np.asarray(a, dtype=np.float64)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def rank_agreement(left, right):
    a, b = ranks(left), ranks(right)
    if a.std() == 0 or b.std() == 0:
        return None  # Undefined, not zero agreement.
    return float(np.corrcoef(a, b)[0, 1])


def validate_records(records, schedule, protocol_hash):
    expected = {(task, row["episode"]): row for task in TASKS for row in schedule}
    found = {}
    for record in records:
        key = (record["task"], record["episode"])
        if key not in expected or key in found:
            raise ValueError("Unexpected/duplicate diagnostic scenario")
        row = expected[key]
        if (record["protocol_sha256"] != protocol_hash or
                any(record[k] != row[k] for k in ("environment_seed", "logical_rank", "local_seed")) or
                set(record["arms"]) != set(ARMS)):
            raise ValueError("Unbound or unpaired diagnostic scenario")
        for name, arm in record["arms"].items():
            summary = score_summary(arm["scores"])
            if arm["selected"] != summary["selected"]:
                raise ValueError("Selection is not the fixed minimum-score rule")
            if (arm["initial_sha256"] != record["initial_sha256"] or
                    arm["physics_sha256"] != record["physics_sha256"] or
                    arm["candidate_sha256"] != record["candidate_sha256"] or
                    arm["goal_sha256"] != record["goal_sha256"] or
                    arm["elementary_steps"] != 15 or
                    not np.isfinite(arm["terminal_ee_distance"]) or arm["terminal_ee_distance"] < 0):
                raise ValueError("Mismatched inputs or incomplete physical prefix")
        found[key] = record
    if set(found) != set(expected):
        raise ValueError(f"Incomplete diagnostic: {len(found)}/{len(expected)} scenarios")
    return found


def analyze(records, schedule, protocol_hash):
    found = validate_records(records, schedule, protocol_hash)
    rng = np.random.default_rng(ANALYSIS["seed"])
    output = {"status": "complete_development_decision_diagnostic_not_full_task_success",
              "analysis": ANALYSIS, "scenarios": len(found), "condition_prefixes": len(found) * 5,
              "protocol_sha256": protocol_hash, "contrasts": [], "rankings": []}
    for task in TASKS:
        rows = [found[task, row["episode"]] for row in schedule]
        # All conditions use the same scenario bootstrap draw, never candidates.
        indices = rng.integers(0, len(rows), size=(ANALYSIS["replicates"], len(rows)))
        for treatment, reference in CONTRASTS:
            delta = np.array([r["arms"][reference]["terminal_ee_distance"] -
                              r["arms"][treatment]["terminal_ee_distance"] for r in rows])
            alpha = (1 - ANALYSIS["confidence"]) / ANALYSIS["family"]
            interval = np.quantile(delta[indices].mean(1), [alpha / 2, 1 - alpha / 2])
            changed = np.array([r["arms"][treatment]["selected"] != r["arms"][reference]["selected"] for r in rows])
            output["contrasts"].append({"task": task, "treatment": treatment, "reference": reference,
                "n_scenarios": len(rows), "mean_distance_gain_m": float(delta.mean()),
                "simultaneous_95_interval_m": interval.tolist(), "changed_choices": int(changed.sum()),
                "changed_only_descriptive_gain_m": float(delta[changed].mean()) if changed.any() else None})
        for name in ARMS:
            agreement = [rank_agreement(r["arms"]["native"]["scores"], r["arms"][name]["scores"]) for r in rows]
            defined = [x for x in agreement if x is not None]
            overlap = [len(set(score_summary(r["arms"]["native"]["scores"])["top10"]) &
                           set(score_summary(r["arms"][name]["scores"])["top10"])) / 10 for r in rows]
            output["rankings"].append({"task": task, "arm": name,
                "mean_spearman": float(np.mean(defined)) if defined else None,
                "undefined_spearman_count": len(rows) - len(defined), "mean_top10_overlap": float(np.mean(overlap)),
                "changed_from_native": sum(r["arms"][name]["selected"] != r["arms"]["native"]["selected"] for r in rows)})
    return output


def main():
    from .protocol import sha256, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    protocol_path = args.root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    digest = sha256(protocol_path)
    if (digest != json.loads((args.root / "FROZEN.json").read_text())["protocol_sha256"] or
            protocol["role"] != ROLE or protocol["analysis"] != ANALYSIS):
        raise ValueError("Diagnostic freeze changed")
    records = []
    for task in TASKS:
        for path in sorted((args.root / task).glob("episode-*/record.json")):
            done = json.loads((path.parent / "DONE.json").read_text())
            for filename, wanted in done["files"].items():
                if Path(filename).name != filename or sha256(path.parent / filename) != wanted:
                    raise ValueError("Diagnostic raw artifact changed")
            records.append(json.loads(path.read_text()))
    result = analyze(records, protocol["episodes"], digest)
    write_json(args.root / "analysis.json", result)
    write_json(args.root / "ANALYSIS_DONE.json", {"analysis_sha256": sha256(args.root / "analysis.json")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
