#!/usr/bin/env python3
"""Fail-closed equivalence and timing gate for Panel-P runtime/cross-host canaries."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np


PAIR_FIELDS = (
    "env",
    "model",
    "task",
    "pair_id",
    "env_seed",
    "planner_seed",
    "checkpoint_sha256",
    "config_sha256",
    "repo_commit",
    "success",
    "n_env_steps",
    "n_plan_calls",
    "executed_action_sha256",
)
REALIZATION_FIELDS = (
    "initial_observation_sha256",
    "goal_observation_sha256",
    "initial_visual_sha256",
    "goal_visual_sha256",
    "initial_proprio_sha256",
    "goal_proprio_sha256",
    "goal_state_sha256",
)


def _load_rows(run_dir: Path, *, require_done: bool) -> list[dict]:
    if require_done and not (run_dir / "DONE.json").is_file():
        raise ValueError(f"candidate has no DONE.json: {run_dir}")
    path = run_dir / "episodes.jsonl"
    if not path.is_file():
        raise ValueError(f"missing episodes.jsonl: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty episodes.jsonl: {path}")
    pair_ids = [str(row.get("pair_id")) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"duplicate pair IDs in {path}")
    return rows


def compare_runs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    min_speedup: float,
    float_atol: float = 0.0,
) -> dict:
    if float_atol < 0:
        raise ValueError("float_atol must be non-negative")
    reference = _load_rows(reference_dir, require_done=False)
    candidate = _load_rows(candidate_dir, require_done=True)
    reference_by_pair = {str(row["pair_id"]): row for row in reference}
    errors: list[str] = []
    compared = []
    max_abs_float_difference = 0.0
    for row in candidate:
        pair_id = str(row["pair_id"])
        if pair_id not in reference_by_pair:
            errors.append(f"candidate pair {pair_id!r} is absent from reference")
            continue
        base = reference_by_pair[pair_id]
        for field in PAIR_FIELDS:
            # A hash cannot express a predeclared numerical tolerance.  When a
            # float tolerance is active, compare the executed-action array below
            # and retain exact checks for every categorical/count field.
            if field == "executed_action_sha256" and float_atol > 0:
                continue
            if row.get(field) != base.get(field):
                errors.append(
                    f"{pair_id}: {field} differs: reference={base.get(field)!r}, "
                    f"candidate={row.get(field)!r}"
                )
        base_realization = base.get("realization_hashes") or {}
        candidate_realization = row.get("realization_hashes") or {}
        for field in REALIZATION_FIELDS:
            if candidate_realization.get(field) != base_realization.get(field):
                errors.append(f"{pair_id}: realization {field} differs")
        base_trace_path = reference_dir / str(base.get("behavior_trace", ""))
        candidate_trace_path = candidate_dir / str(row.get("behavior_trace", ""))
        if not base_trace_path.is_file() or not candidate_trace_path.is_file():
            errors.append(f"{pair_id}: behavior trace missing")
        else:
            with np.load(base_trace_path) as base_trace, np.load(candidate_trace_path) as candidate_trace:
                if set(base_trace.files) != set(candidate_trace.files):
                    errors.append(f"{pair_id}: behavior trace keys differ")
                for key in sorted(set(base_trace.files).intersection(candidate_trace.files)):
                    left, right = base_trace[key], candidate_trace[key]
                    if left.shape != right.shape or left.dtype != right.dtype:
                        errors.append(
                            f"{pair_id}: behavior trace array {key!r} shape/dtype differs"
                        )
                        continue
                    floating = np.issubdtype(left.dtype, np.floating)
                    if floating and float_atol > 0:
                        finite = np.isfinite(left) & np.isfinite(right)
                        if finite.any():
                            max_abs_float_difference = max(
                                max_abs_float_difference,
                                float(np.max(np.abs(left[finite] - right[finite]))),
                            )
                        equal = np.allclose(
                            left,
                            right,
                            rtol=0.0,
                            atol=float_atol,
                            equal_nan=True,
                        )
                    else:
                        equal = np.array_equal(left, right, equal_nan=True)
                    if not equal:
                        errors.append(f"{pair_id}: behavior trace array {key!r} differs")
        compared.append((base, row))
    reference_mean_s = sum(float(a["wall_s"]) for a, _ in compared) / max(len(compared), 1)
    candidate_mean_s = sum(float(b["wall_s"]) for _, b in compared) / max(len(compared), 1)
    speedup = reference_mean_s / candidate_mean_s if candidate_mean_s > 0 else 0.0
    if speedup < min_speedup:
        errors.append(f"speedup {speedup:.4f} is below required {min_speedup:.4f}")
    return {
        "schema_version": "panel-p-runtime-canary-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_dir": str(reference_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "candidate_pairs": [str(row["pair_id"]) for row in candidate],
        "n_compared": len(compared),
        "exact_fields": [
            field for field in PAIR_FIELDS
            if not (field == "executed_action_sha256" and float_atol > 0)
        ],
        "float_atol": float_atol,
        "max_abs_float_difference": max_abs_float_difference,
        "action_comparison": (
            "executed_actions array with rtol=0 and float_atol"
            if float_atol > 0 else "executed_action_sha256 and array exact"
        ),
        "exact_realization_fields": list(REALIZATION_FIELDS),
        "behavior_trace_comparison": "array keys/dtypes/shapes/values exact; NPZ byte hash excluded",
        "reference_mean_wall_s": reference_mean_s,
        "candidate_mean_wall_s": candidate_mean_s,
        "speedup": speedup,
        "min_speedup": min_speedup,
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="use 1.20 for the visualization-free gate and 0 for equivalence-only host checks",
    )
    ap.add_argument(
        "--float-atol",
        type=float,
        default=0.0,
        help="absolute tolerance for floating trace arrays; integers and categorical fields remain exact",
    )
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite canary report: {args.out}")
    try:
        report = compare_runs(
            args.reference_dir,
            args.candidate_dir,
            min_speedup=args.min_speedup,
            float_atol=args.float_atol,
        )
    except Exception as exc:
        report = {
            "schema_version": "panel-p-runtime-canary-v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reference_dir": str(args.reference_dir.resolve()),
            "candidate_dir": str(args.candidate_dir.resolve()),
            "passed": False,
            "errors": [str(exc)],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
