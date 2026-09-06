#!/usr/bin/env python3
"""Fail-closed P0 behavioral-substrate gate for manifest-driven Panel P runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from public_panel_manifest import read_jsonl, rows_sha256


HASH_FIELDS = (
    "initial_observation_sha256",
    "goal_observation_sha256",
    "initial_visual_sha256",
    "goal_visual_sha256",
    "initial_proprio_sha256",
    "goal_proprio_sha256",
    "goal_state_sha256",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - radius, center + radius


def evaluate(
    run_dir: Path,
    manifest_path: Path,
    *,
    min_success: int = 6,
    min_failure: int = 6,
    min_rate: float = 0.20,
    max_rate: float = 0.80,
    min_plan_calls: int = 2,
) -> dict:
    manifest = read_jsonl(manifest_path)
    episode_path = run_dir / "episodes.jsonl"
    done_path = run_dir / "DONE.json"
    errors: list[str] = []
    if not episode_path.exists():
        errors.append("missing episodes.jsonl")
        episodes = []
    else:
        episodes = read_jsonl(episode_path)
    if not done_path.exists():
        errors.append("missing DONE.json")
        done = {}
    else:
        done = json.loads(done_path.read_text())

    if len(episodes) != len(manifest):
        errors.append(f"episode count {len(episodes)} != manifest count {len(manifest)}")
    if not done.get("complete", False):
        errors.append("runner did not declare complete")
    if int(done.get("n_logged", -1)) != len(manifest):
        errors.append("DONE n_logged does not match manifest")

    manifest_sha = rows_sha256(manifest)
    expected_pairs = [str(row["pair_id"]) for row in manifest]
    actual_pairs = [str(row.get("pair_id")) for row in episodes]
    if actual_pairs != expected_pairs:
        errors.append("episode pair IDs/order do not exactly match manifest")

    for index, (expected, actual) in enumerate(zip(manifest, episodes)):
        for field in ("task", "env_seed", "planner_seed", "checkpoint_sha256", "config_sha256"):
            if str(actual.get(field)) != str(expected.get(field)):
                errors.append(f"episode {index}: {field} differs from manifest")
        if actual.get("episode_manifest_sha256") != manifest_sha:
            errors.append(f"episode {index}: manifest SHA differs")
        hashes = actual.get("realization_hashes") or {}
        missing = [field for field in HASH_FIELDS if not hashes.get(field)]
        if missing:
            errors.append(f"episode {index}: missing realization hashes {missing}")
        calls = actual.get("n_plan_calls")
        if calls is None or int(calls) < min_plan_calls:
            errors.append(f"episode {index}: n_plan_calls={calls!r} below {min_plan_calls}")

    outcomes = [int(bool(row.get("success"))) for row in episodes]
    n = len(outcomes)
    successes = sum(outcomes)
    failures = n - successes
    rate = successes / n if n else math.nan
    if successes < min_success:
        errors.append(f"success class floor failed: {successes} < {min_success}")
    if failures < min_failure:
        errors.append(f"failure class floor failed: {failures} < {min_failure}")
    if n and not min_rate <= rate <= max_rate:
        errors.append(f"operating-point band failed: {rate:.6f} not in [{min_rate}, {max_rate}]")

    quartile_success = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4
        quartile_success.append(sum(outcomes[lo:hi]))
    success_quartiles = sum(value > 0 for value in quartile_success)
    failure_quartiles = sum((hi - lo - quartile_success[q]) > 0 for q, (lo, hi) in enumerate(
        ((q * n // 4, (q + 1) * n // 4) for q in range(4))
    ))
    if n and (success_quartiles < 2 or failure_quartiles < 2):
        errors.append("one outcome class is confined to fewer than two ordinal quartiles")

    interval = wilson_interval(successes, n)
    return {
        "passed": not errors,
        "decision": "advance_to_fit" if not errors else "stop_substrate",
        "run_dir": str(run_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "n": n,
        "successes": successes,
        "failures": failures,
        "success_rate": rate,
        "wilson_95": list(interval),
        "quartile_successes": quartile_success,
        "success_quartiles": success_quartiles,
        "failure_quartiles": failure_quartiles,
        "thresholds": {
            "min_success": min_success,
            "min_failure": min_failure,
            "min_rate": min_rate,
            "max_rate": max_rate,
            "min_plan_calls": min_plan_calls,
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-success", type=int, default=6)
    parser.add_argument("--min-failure", type=int, default=6)
    parser.add_argument("--min-rate", type=float, default=0.20)
    parser.add_argument("--max-rate", type=float, default=0.80)
    parser.add_argument("--min-plan-calls", type=int, default=2)
    args = parser.parse_args()
    result = evaluate(
        args.run_dir,
        args.manifest,
        min_success=args.min_success,
        min_failure=args.min_failure,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        min_plan_calls=args.min_plan_calls,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
