#!/usr/bin/env python3
"""Fail-closed paired development gate for Panel-P intervention arms.

This command consumes completed *development* runs only.  It never opens a
protected manifest and never chooses an arm.  The caller must name the selected
arm and any dose-matched or static/HMM comparisons that were frozen for the
development tournament.

The gate proves four separate things before ``public_panel_arm_registry.py`` can
freeze a protected arm set:

1. identity reproduces unsteered actions and outcomes bit-for-bit;
2. every treatment changes actions, stays below the cap-saturation ceiling, and
   satisfies declared action-dose matches;
3. an HMM arm, when requested, is a meaningfully different treatment from its
   static counterpart;
4. the explicitly selected utility arm clears a permissive development-only
   promotion threshold.  Protected efficacy is still decided on unused pairs.

Every comparison is paired by immutable ``pair_id`` and requires identical
environment/planner seeds, realization hashes, checkpoint/config identities,
episode order, action shapes, and plan boundaries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "panel-p-development-gate-v1"
PAIR_FIELDS = (
    "task",
    "split",
    "pair_id",
    "env_seed",
    "planner_seed",
    "checkpoint_sha256",
    "config_sha256",
    "repo_commit",
    "episode_manifest_sha256",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing episode log: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty episode log: {path}")
    return rows


@dataclass(frozen=True)
class CompletedRun:
    path: Path
    done: dict
    rows: tuple[dict, ...]
    actions: tuple[np.ndarray, ...]
    boundaries: tuple[np.ndarray, ...]


def _trace_path(run_dir: Path, row: dict) -> Path:
    raw = row.get("behavior_trace")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"pair {row.get('pair_id')!r} has no behavior_trace")
    path = Path(raw)
    return path if path.is_absolute() else run_dir / path


def load_completed_run(path: Path) -> CompletedRun:
    path = path.expanduser().resolve()
    done_path = path / "DONE.json"
    if not done_path.is_file():
        raise ValueError(f"run is not complete (missing DONE.json): {path}")
    if (path / "INCOMPLETE.json").exists():
        raise ValueError(f"run contains INCOMPLETE.json: {path}")
    done = json.loads(done_path.read_text())
    if done.get("complete") is not True:
        raise ValueError(f"run DONE.json is not complete: {path}")

    rows = load_jsonl(path / "episodes.jsonl")
    expected = int(done.get("expected_episodes", done.get("n_logged", -1)))
    if int(done.get("n_logged", -1)) != len(rows) or expected != len(rows):
        raise ValueError(
            f"run completion count mismatch at {path}: "
            f"rows={len(rows)} n_logged={done.get('n_logged')} expected={expected}"
        )
    pair_ids = [row.get("pair_id") for row in rows]
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError(f"run contains a missing pair_id: {path}")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"run contains duplicate pair_id values: {path}")
    if {str(row.get("split")) for row in rows} != {"development"}:
        raise ValueError(f"development gate received a non-development run: {path}")

    actions: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    for row in rows:
        trace_path = _trace_path(path, row)
        if not trace_path.is_file():
            raise ValueError(f"missing behavior trace for {row['pair_id']}: {trace_path}")
        expected_sha = row.get("behavior_trace_sha256")
        if expected_sha and sha256_file(trace_path) != expected_sha:
            raise ValueError(f"behavior trace hash mismatch for {row['pair_id']}: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as trace:
            if not {"executed_actions", "plan_boundaries"}.issubset(trace.files):
                raise ValueError(f"trace lacks executed_actions/plan_boundaries: {trace_path}")
            action = np.asarray(trace["executed_actions"])
            boundary = np.asarray(trace["plan_boundaries"])
        if action.ndim != 2 or not len(action) or not np.isfinite(action).all():
            raise ValueError(f"invalid executed_actions in {trace_path}: shape={action.shape}")
        if boundary.ndim != 1 or len(boundary) < 2 or int(boundary[0]) != 0 or int(boundary[-1]) != len(action):
            raise ValueError(f"invalid plan_boundaries in {trace_path}")
        if np.any(np.diff(boundary) <= 0):
            raise ValueError(f"non-increasing plan_boundaries in {trace_path}")
        actions.append(action)
        boundaries.append(boundary)
    return CompletedRun(path, done, tuple(rows), tuple(actions), tuple(boundaries))


def _require_paired(reference: CompletedRun, candidate: CompletedRun, name: str) -> None:
    if len(reference.rows) != len(candidate.rows):
        raise ValueError(f"{name} has {len(candidate.rows)} rows; baseline has {len(reference.rows)}")
    for index, (left, right) in enumerate(zip(reference.rows, candidate.rows)):
        for field in PAIR_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(
                    f"{name} pair index {index} field {field!r} differs: "
                    f"{left.get(field)!r} != {right.get(field)!r}"
                )
        if left.get("realization_hashes") != right.get("realization_hashes"):
            raise ValueError(f"{name} realization hashes differ for pair {left.get('pair_id')}")
        if reference.actions[index].shape != candidate.actions[index].shape:
            raise ValueError(f"{name} action shape differs for pair {left.get('pair_id')}")
        if not np.array_equal(reference.boundaries[index], candidate.boundaries[index]):
            raise ValueError(f"{name} plan boundaries differ for pair {left.get('pair_id')}")


def _bootstrap_ci(values: np.ndarray, *, draws: int, seed: int, alpha: float = 0.05) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    # Batched sampling avoids allocating a potentially large draws-by-episode array.
    batch = 1000
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def _action_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rms_coordinate = float(np.sqrt(np.mean(delta * delta)))
    mean_step_l2 = float(np.linalg.norm(delta, axis=1).mean())
    return rms_coordinate, mean_step_l2


def _arm_summary(
    baseline: CompletedRun,
    arm: CompletedRun,
    *,
    draws: int,
    seed: int,
    minimum_action_distance: float,
    cap_saturation_ceiling: float,
) -> dict:
    rms = []
    step_l2 = []
    success_delta = []
    changed_hashes = 0
    for index, (base_row, arm_row) in enumerate(zip(baseline.rows, arm.rows)):
        r, l2 = _action_distance(baseline.actions[index], arm.actions[index])
        rms.append(r)
        step_l2.append(l2)
        success_delta.append(int(bool(arm_row.get("success"))) - int(bool(base_row.get("success"))))
        changed_hashes += int(base_row.get("executed_action_sha256") != arm_row.get("executed_action_sha256"))
    rms_array = np.asarray(rms)
    step_array = np.asarray(step_l2)
    outcome_array = np.asarray(success_delta, dtype=float)

    runtime_arm = str(arm.done.get("arm", ""))
    cap_required = runtime_arm.startswith("sonar_")
    cap_value = arm.done.get("cap_hit_fraction_supported")
    cap_present = isinstance(cap_value, (int, float)) and np.isfinite(float(cap_value))
    cap_passed = (not cap_required) or (cap_present and float(cap_value) <= cap_saturation_ceiling)
    nonzero = bool(float(step_array.mean()) > minimum_action_distance and changed_hashes > 0)
    return {
        "n_pairs": len(rms),
        "runtime_arm": runtime_arm or None,
        "mean_action_rms_coordinate": float(rms_array.mean()),
        "action_rms_coordinate_ci95": _bootstrap_ci(rms_array, draws=draws, seed=seed),
        "mean_action_step_l2": float(step_array.mean()),
        "action_step_l2_ci95": _bootstrap_ci(step_array, draws=draws, seed=seed + 1),
        "changed_action_hash_pairs": int(changed_hashes),
        "changed_action_hash_fraction": float(changed_hashes / len(rms)),
        "nonzero_treatment": nonzero,
        "cap_saturation_required": cap_required,
        "cap_hit_fraction_supported": float(cap_value) if cap_present else None,
        "cap_saturation_ceiling": float(cap_saturation_ceiling),
        "cap_saturation_passed": bool(cap_passed),
        "paired_success_difference": float(outcome_array.mean()),
        "paired_success_difference_ci95": _bootstrap_ci(outcome_array, draws=draws, seed=seed + 2),
        "baseline_fail_arm_success": int(np.sum(outcome_array == 1)),
        "baseline_success_arm_fail": int(np.sum(outcome_array == -1)),
        "passed": bool(nonzero and cap_passed),
    }


def _identity_report(baseline: CompletedRun, identity: CompletedRun) -> dict:
    mismatches = []
    for index, (base_row, identity_row) in enumerate(zip(baseline.rows, identity.rows)):
        pair_id = str(base_row["pair_id"])
        if bool(base_row.get("success")) != bool(identity_row.get("success")):
            mismatches.append({"pair_id": pair_id, "field": "success"})
        if base_row.get("executed_action_sha256") != identity_row.get("executed_action_sha256"):
            mismatches.append({"pair_id": pair_id, "field": "executed_action_sha256"})
        if not np.array_equal(baseline.actions[index], identity.actions[index]):
            mismatches.append({"pair_id": pair_id, "field": "executed_actions"})
        if not np.array_equal(baseline.boundaries[index], identity.boundaries[index]):
            mismatches.append({"pair_id": pair_id, "field": "plan_boundaries"})
    return {
        "schema_version": SCHEMA_VERSION,
        "n_pairs": len(baseline.rows),
        "required_equality": "bit-exact outcomes, action hashes, flattened actions, and plan boundaries",
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _treatment_separation_report(
    baseline: CompletedRun,
    static: CompletedRun | None,
    hmm: CompletedRun | None,
    *,
    draws: int,
    seed: int,
    minimum_mean: float,
    minimum_ci_lower: float,
    minimum_changed_fraction: float,
) -> dict:
    report = {
        "schema_version": SCHEMA_VERSION,
        "applicable": static is not None or hmm is not None,
        "thresholds": {
            "minimum_mean_normalized_separation": minimum_mean,
            "minimum_ci95_lower": minimum_ci_lower,
            "minimum_changed_action_hash_fraction": minimum_changed_fraction,
        },
    }
    if static is None or hmm is None:
        report.update({"passed": False, "reason": "static_and_hmm_runs_not_both_declared"})
        return report

    normalized = []
    changed = 0
    for index, (static_row, hmm_row) in enumerate(zip(static.rows, hmm.rows)):
        _, between = _action_distance(static.actions[index], hmm.actions[index])
        _, static_dose = _action_distance(static.actions[index], baseline.actions[index])
        _, hmm_dose = _action_distance(hmm.actions[index], baseline.actions[index])
        normalized.append(between / (0.5 * (static_dose + hmm_dose) + 1e-12))
        changed += int(static_row.get("executed_action_sha256") != hmm_row.get("executed_action_sha256"))
    values = np.asarray(normalized, dtype=float)
    ci = _bootstrap_ci(values, draws=draws, seed=seed)
    changed_fraction = float(changed / len(values))
    passed = bool(
        float(values.mean()) >= minimum_mean
        and ci[0] > minimum_ci_lower
        and changed_fraction >= minimum_changed_fraction
    )
    report.update({
        "n_pairs": int(len(values)),
        "mean_normalized_separation": float(values.mean()),
        "normalized_separation_ci95": ci,
        "changed_action_hash_pairs": int(changed),
        "changed_action_hash_fraction": changed_fraction,
        "passed": passed,
    })
    return report


def _write_exclusive(path: Path, document: dict) -> None:
    with path.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate_development(
    baseline_path: Path,
    identity_path: Path,
    arm_paths: dict[str, Path],
    dose_pairs: list[tuple[str, str]],
    selected_arm: str,
    out: Path,
    *,
    static_arm: str | None = None,
    hmm_arm: str | None = None,
    bootstrap_draws: int = 10000,
    seed: int = 0,
    minimum_action_distance: float = 1e-6,
    cap_saturation_ceiling: float = 0.25,
    dose_ratio_low: float = 0.75,
    dose_ratio_high: float = 1.25,
    minimum_separation_mean: float = 0.25,
    minimum_separation_ci_lower: float = 0.10,
    minimum_separation_changed_fraction: float = 0.25,
    minimum_selected_gain: float = 0.05,
    minimum_selected_ci_lower: float = -0.10,
) -> dict:
    if not arm_paths:
        raise ValueError("at least one treated development arm is required")
    if len(arm_paths) != len(set(arm_paths)) or any(not key for key in arm_paths):
        raise ValueError("treated arm ids must be unique nonempty strings")
    if selected_arm not in arm_paths:
        raise ValueError(f"selected arm {selected_arm!r} is not one of {sorted(arm_paths)}")
    if not dose_pairs:
        raise ValueError("at least one preregistered action-dose pair is required")
    known = set(arm_paths)
    dose_covered: set[str] = set()
    for left, right in dose_pairs:
        if left not in known or right not in known or left == right:
            raise ValueError(f"invalid dose pair {left!r}={right!r}")
        dose_covered.update((left, right))
    missing_dose_matches = sorted(known - dose_covered)
    if missing_dose_matches:
        raise ValueError(f"treated arms lack a declared action-dose match: {missing_dose_matches}")
    if (static_arm is None) != (hmm_arm is None):
        raise ValueError("--static-arm and --hmm-arm must be declared together")
    if static_arm is not None and (static_arm not in known or hmm_arm not in known):
        raise ValueError("static/HMM arm ids must refer to declared treated arms")
    if not 0 < dose_ratio_low <= 1 <= dose_ratio_high:
        raise ValueError("dose ratio bounds must straddle one")

    baseline = load_completed_run(baseline_path)
    identity = load_completed_run(identity_path)
    _require_paired(baseline, identity, "identity")
    arms = {name: load_completed_run(path) for name, path in arm_paths.items()}
    for name, run in arms.items():
        _require_paired(baseline, run, name)

    identity_report = _identity_report(baseline, identity)
    arm_reports = {
        name: _arm_summary(
            baseline,
            run,
            draws=bootstrap_draws,
            seed=seed + 100 * index,
            minimum_action_distance=minimum_action_distance,
            cap_saturation_ceiling=cap_saturation_ceiling,
        )
        for index, (name, run) in enumerate(sorted(arms.items()))
    }

    pair_reports = []
    for left, right in dose_pairs:
        left_dose = arm_reports[left]["mean_action_step_l2"]
        right_dose = arm_reports[right]["mean_action_step_l2"]
        ratio = float(left_dose / (right_dose + 1e-12))
        pair_reports.append({
            "left": left,
            "right": right,
            "left_over_right": ratio,
            "bounds": [dose_ratio_low, dose_ratio_high],
            "passed": bool(dose_ratio_low <= ratio <= dose_ratio_high),
        })
    action_dose_report = {
        "schema_version": SCHEMA_VERSION,
        "minimum_action_distance": minimum_action_distance,
        "arms": arm_reports,
        "dose_pairs": pair_reports,
        "passed": bool(all(value["passed"] for value in arm_reports.values()) and all(value["passed"] for value in pair_reports)),
    }

    separation_report = _treatment_separation_report(
        baseline,
        arms.get(static_arm) if static_arm else None,
        arms.get(hmm_arm) if hmm_arm else None,
        draws=bootstrap_draws,
        seed=seed + 9001,
        minimum_mean=minimum_separation_mean,
        minimum_ci_lower=minimum_separation_ci_lower,
        minimum_changed_fraction=minimum_separation_changed_fraction,
    )

    selected = arm_reports[selected_arm]
    selected_ci = selected["paired_success_difference_ci95"]
    utility_passed = bool(
        selected["passed"]
        and selected["paired_success_difference"] >= minimum_selected_gain
        and selected_ci[0] >= minimum_selected_ci_lower
    )
    utility_report = {
        "schema_version": SCHEMA_VERSION,
        "selected_arm": selected_arm,
        "selection_was_supplied_not_optimized_by_gate": True,
        "paired_success_difference": selected["paired_success_difference"],
        "paired_success_difference_ci95": selected_ci,
        "minimum_selected_gain": minimum_selected_gain,
        "minimum_selected_ci_lower": minimum_selected_ci_lower,
        "protected_efficacy_not_inferred_here": True,
        "passed": utility_passed,
    }

    hmm_required = static_arm is not None
    overall = bool(
        identity_report["passed"]
        and action_dose_report["passed"]
        and utility_report["passed"]
        and (separation_report["passed"] if hmm_required else True)
    )
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_names = (
        "identity_action_hash.json",
        "action_dose.json",
        "treatment_separation.json",
        "development_utility.json",
        "development_gate.json",
    )
    existing = [str(out / name) for name in output_names if (out / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite development evidence: {existing}")

    provenance = {
        "baseline": {"path": str(baseline.path), "done_sha256": sha256_file(baseline.path / "DONE.json"), "episodes_sha256": sha256_file(baseline.path / "episodes.jsonl")},
        "identity": {"path": str(identity.path), "done_sha256": sha256_file(identity.path / "DONE.json"), "episodes_sha256": sha256_file(identity.path / "episodes.jsonl")},
        "arms": {
            name: {"path": str(run.path), "done_sha256": sha256_file(run.path / "DONE.json"), "episodes_sha256": sha256_file(run.path / "episodes.jsonl")}
            for name, run in sorted(arms.items())
        },
    }
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    for report in (identity_report, action_dose_report, separation_report, utility_report):
        report["generated_utc"] = generated
        report["input_provenance"] = provenance
    gate_report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "input_provenance": provenance,
        "selected_arm": selected_arm,
        "hmm_separation_required": hmm_required,
        "checks": {
            "identity_action_hash": bool(identity_report["passed"]),
            "action_dose": bool(action_dose_report["passed"]),
            "development_utility": bool(utility_report["passed"]),
            "treatment_separation": bool(separation_report["passed"]) if hmm_required else "not_applicable",
        },
        "passed": overall,
        "decision": "eligible_for_arm_registry_review" if overall else "stop_before_arm_registry",
    }

    _write_exclusive(out / "identity_action_hash.json", identity_report)
    _write_exclusive(out / "action_dose.json", action_dose_report)
    _write_exclusive(out / "treatment_separation.json", separation_report)
    _write_exclusive(out / "development_utility.json", utility_report)
    _write_exclusive(out / "development_gate.json", gate_report)
    return gate_report


def _mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ID=PATH")
    key, raw = value.split("=", 1)
    if not key or not raw:
        raise argparse.ArgumentTypeError("expected nonempty ID=PATH")
    return key, Path(raw)


def _pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LEFT=RIGHT")
    left, right = value.split("=", 1)
    if not left or not right:
        raise argparse.ArgumentTypeError("expected nonempty LEFT=RIGHT")
    return left, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--arm", required=True, action="append", type=_mapping, metavar="ID=PATH")
    parser.add_argument("--dose-match", required=True, action="append", type=_pair, metavar="LEFT=RIGHT")
    parser.add_argument("--selected-arm", required=True)
    parser.add_argument("--static-arm")
    parser.add_argument("--hmm-arm")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    arm_paths = dict(args.arm)
    if len(arm_paths) != len(args.arm):
        raise SystemExit("duplicate --arm id")
    report = evaluate_development(
        args.baseline,
        args.identity,
        arm_paths,
        args.dose_match,
        args.selected_arm,
        args.out,
        static_arm=args.static_arm,
        hmm_arm=args.hmm_arm,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
