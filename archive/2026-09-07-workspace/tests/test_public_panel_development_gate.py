from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_development_gate import evaluate_development, sha256_file  # noqa: E402


def _action_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_run(
    root: Path,
    name: str,
    actions: list[np.ndarray],
    outcomes: list[int],
    *,
    runtime_arm: str | None,
    cap_fraction: float | None = 0.1,
    complete: bool = True,
) -> Path:
    run = root / name
    traces = run / "behavior_traces"
    traces.mkdir(parents=True)
    rows = []
    for index, (action, outcome) in enumerate(zip(actions, outcomes)):
        trace = traces / f"ep{index:03d}.npz"
        np.savez_compressed(
            trace,
            executed_actions=np.asarray(action, dtype=np.float32),
            plan_boundaries=np.asarray([0, 2, 4], dtype=np.int32),
        )
        rows.append({
            "task": "mw-reach",
            "split": "development",
            "pair_id": f"development-{index:03d}",
            "env_seed": 100 + index,
            "planner_seed": 200 + index,
            "checkpoint_sha256": "c" * 64,
            "config_sha256": "f" * 64,
            "repo_commit": "1" * 40,
            "episode_manifest_sha256": "m" * 64,
            "realization_hashes": {"initial": f"i{index}", "goal": f"g{index}"},
            "success": int(outcome),
            "executed_action_sha256": _action_hash(np.asarray(action, dtype=np.float32)),
            "behavior_trace": str(trace.relative_to(run)),
            "behavior_trace_sha256": sha256_file(trace),
        })
    (run / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    done = {
        "complete": complete,
        "n_logged": len(rows),
        "expected_episodes": len(rows),
    }
    if runtime_arm is not None:
        done["arm"] = runtime_arm
    if cap_fraction is not None:
        done["cap_hit_fraction_supported"] = cap_fraction
    (run / ("DONE.json" if complete else "INCOMPLETE.json")).write_text(json.dumps(done))
    return run


def _fixture(tmp_path: Path, *, sham_scale: float = 1.0, hmm_matches_static: bool = False):
    n = 10
    baseline_actions = [np.zeros((4, 2), dtype=np.float32) for _ in range(n)]
    static_actions = [np.tile([[1.0, 0.0]], (4, 1)).astype(np.float32) for _ in range(n)]
    sham_actions = [np.tile([[0.0, sham_scale]], (4, 1)).astype(np.float32) for _ in range(n)]
    hmm_actions = [value.copy() for value in static_actions] if hmm_matches_static else [
        np.tile([[-1.0, 0.0]], (4, 1)).astype(np.float32) for _ in range(n)
    ]
    baseline_outcomes = [0, 1] * 5
    selected_outcomes = baseline_outcomes.copy()
    selected_outcomes[0] = 1
    selected_outcomes[2] = 1

    baseline = _write_run(tmp_path, "baseline", baseline_actions, baseline_outcomes, runtime_arm=None, cap_fraction=None)
    identity = _write_run(tmp_path, "identity", baseline_actions, baseline_outcomes, runtime_arm="identity", cap_fraction=None)
    static = _write_run(tmp_path, "static", static_actions, selected_outcomes, runtime_arm="sonar_coast_static")
    sham = _write_run(tmp_path, "sham", sham_actions, baseline_outcomes, runtime_arm="sonar_coast_sham")
    hmm = _write_run(tmp_path, "hmm", hmm_actions, selected_outcomes, runtime_arm="sonar_coast_hmm")
    return baseline, identity, {"static": static, "sham": sham, "hmm": hmm}


def test_development_gate_passes_paired_identity_dose_utility_and_hmm(tmp_path: Path) -> None:
    baseline, identity, arms = _fixture(tmp_path)
    report = evaluate_development(
        baseline,
        identity,
        arms,
        [("static", "sham"), ("hmm", "static")],
        "static",
        tmp_path / "evidence",
        static_arm="static",
        hmm_arm="hmm",
        bootstrap_draws=1000,
    )
    assert report["passed"]
    evidence = tmp_path / "evidence"
    assert json.loads((evidence / "identity_action_hash.json").read_text())["passed"]
    assert json.loads((evidence / "action_dose.json").read_text())["passed"]
    assert json.loads((evidence / "treatment_separation.json").read_text())["passed"]
    assert json.loads((evidence / "development_utility.json").read_text())["passed"]


def test_development_gate_fails_nonidentity_actions(tmp_path: Path) -> None:
    baseline, identity, arms = _fixture(tmp_path)
    trace = identity / "behavior_traces" / "ep000.npz"
    np.savez_compressed(
        trace,
        executed_actions=np.ones((4, 2), dtype=np.float32),
        plan_boundaries=np.asarray([0, 2, 4], dtype=np.int32),
    )
    rows = [json.loads(line) for line in (identity / "episodes.jsonl").read_text().splitlines()]
    rows[0]["behavior_trace_sha256"] = sha256_file(trace)
    rows[0]["executed_action_sha256"] = _action_hash(np.ones((4, 2), dtype=np.float32))
    (identity / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = evaluate_development(
        baseline,
        identity,
        arms,
        [("static", "sham"), ("hmm", "static")],
        "static",
        tmp_path / "evidence",
        static_arm="static",
        hmm_arm="hmm",
        bootstrap_draws=500,
    )
    assert not report["passed"]
    assert not report["checks"]["identity_action_hash"]


def test_development_gate_fails_action_dose_mismatch(tmp_path: Path) -> None:
    baseline, identity, arms = _fixture(tmp_path, sham_scale=0.2)
    report = evaluate_development(
        baseline,
        identity,
        arms,
        [("static", "sham"), ("hmm", "static")],
        "static",
        tmp_path / "evidence",
        static_arm="static",
        hmm_arm="hmm",
        bootstrap_draws=500,
    )
    assert not report["passed"]
    dose = json.loads((tmp_path / "evidence" / "action_dose.json").read_text())
    assert not dose["passed"]
    assert not dose["dose_pairs"][0]["passed"]


def test_development_gate_fails_static_hmm_treatment_aliasing(tmp_path: Path) -> None:
    baseline, identity, arms = _fixture(tmp_path, hmm_matches_static=True)
    report = evaluate_development(
        baseline,
        identity,
        arms,
        [("static", "sham"), ("hmm", "static")],
        "static",
        tmp_path / "evidence",
        static_arm="static",
        hmm_arm="hmm",
        bootstrap_draws=500,
    )
    assert not report["passed"]
    assert not report["checks"]["treatment_separation"]


def test_development_gate_rejects_incomplete_run(tmp_path: Path) -> None:
    baseline, identity, arms = _fixture(tmp_path)
    (arms["static"] / "DONE.json").unlink()
    with pytest.raises(ValueError, match="missing DONE.json"):
        evaluate_development(
            baseline,
            identity,
            arms,
            [("static", "sham"), ("hmm", "static")],
            "static",
            tmp_path / "evidence",
            static_arm="static",
            hmm_arm="hmm",
            bootstrap_draws=100,
        )
