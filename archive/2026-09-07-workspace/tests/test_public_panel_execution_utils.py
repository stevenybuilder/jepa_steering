from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "cgs_pilot"
sys.path.insert(0, str(SCRIPTS))

from public_panel_eval import apply_optional_visualization_policy  # noqa: E402
from public_panel_manifest import SeedAllocator, generate_rows, validate_rows  # noqa: E402
from public_panel_shard_manifest import shard_rows  # noqa: E402
from public_panel_runtime_canary import compare_runs  # noqa: E402


def _rows(n: int = 30) -> list[dict]:
    return generate_rows(
        split="fit",
        count=n,
        task="mw-reach",
        namespace="test-sharding",
        checkpoint_sha256="c" * 64,
        config_sha256="f" * 64,
        repo_commit="a" * 40,
        allocator=SeedAllocator("test-sharding"),
    )


def test_optional_visualization_policy_is_explicit_and_idempotent():
    params = {
        "planner": {"decode_each_iteration": True, "iterations": 15},
        "logging": {"optional_plots": True},
    }
    deviations: list[dict] = []
    apply_optional_visualization_policy(params, deviations, disabled=True)
    assert params["planner"]["decode_each_iteration"] is False
    assert params["logging"]["optional_plots"] is False
    assert params["planner"]["iterations"] == 15
    assert [item["field"] for item in deviations] == [
        "planner.decode_each_iteration",
        "logging.optional_plots",
    ]
    apply_optional_visualization_policy(params, deviations, disabled=True)
    assert len(deviations) == 2


def test_six_shards_are_valid_disjoint_and_cover_source():
    rows = _rows()
    shards = [shard_rows(rows, index=i, count=6) for i in range(6)]
    assert [len(shard) for shard in shards] == [5] * 6
    assert all(validate_rows((("shard", shard),))["passed"] for shard in shards)
    pair_sets = [{row["pair_id"] for row in shard} for shard in shards]
    assert set.union(*pair_sets) == {row["pair_id"] for row in rows}
    assert sum(len(pair_set) for pair_set in pair_sets) == len(set.union(*pair_sets))
    for shard in shards:
        assert [row["ordinal"] for row in shard] == list(range(len(shard)))
        assert all(row["pair_id"].startswith("fit-") for row in shard)


def test_sharding_rejects_invalid_coordinates():
    rows = _rows(3)
    for index, count in ((-1, 2), (2, 2), (0, 0)):
        try:
            shard_rows(rows, index=index, count=count)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid shard coordinates {index}/{count}")


def _canary_row(pair_id: str, wall_s: float = 10.0) -> dict:
    return {
        "env": "mw",
        "model": "dino-wm",
        "task": "mw-reach",
        "pair_id": pair_id,
        "env_seed": 1,
        "planner_seed": 2,
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "f" * 64,
        "repo_commit": "a" * 40,
        "success": 1,
        "n_env_steps": 99,
        "n_plan_calls": 7,
        "executed_action_sha256": "e" * 64,
        "behavior_trace_sha256": "b" * 64,
        "wall_s": wall_s,
        "realization_hashes": {field: field + "-hash" for field in (
            "initial_observation_sha256",
            "goal_observation_sha256",
            "initial_visual_sha256",
            "goal_visual_sha256",
            "initial_proprio_sha256",
            "goal_proprio_sha256",
            "goal_state_sha256",
        )},
    }


def _write_canary_run(path: Path, rows: list[dict], *, done: bool) -> None:
    path.mkdir(parents=True)
    trace_dir = path / "behavior_traces"
    trace_dir.mkdir()
    rendered = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["behavior_trace"] = f"behavior_traces/ep{index:03d}.npz"
        np.savez_compressed(
            path / row["behavior_trace"],
            executed_actions=np.asarray([[1.0, 2.0]], dtype=np.float32),
            plan_boundaries=np.asarray([0, 1], dtype=np.int32),
        )
        rendered.append(row)
    (path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rendered))
    if done:
        (path / "DONE.json").write_text(json.dumps({"complete": True}))


def test_runtime_canary_requires_exact_behavior_and_speed(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canary_run(reference, [_canary_row("development-000", 10.0)], done=False)
    _write_canary_run(candidate, [_canary_row("development-000", 5.0)], done=True)
    report = compare_runs(reference, candidate, min_speedup=1.2)
    assert report["passed"]
    assert report["speedup"] == 2.0


def test_runtime_canary_fails_on_action_difference(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canary_run(reference, [_canary_row("development-000")], done=False)
    changed = _canary_row("development-000")
    changed["executed_action_sha256"] = "x" * 64
    _write_canary_run(candidate, [changed], done=True)
    report = compare_runs(reference, candidate, min_speedup=0.0)
    assert not report["passed"]
    assert any("executed_action_sha256 differs" in error for error in report["errors"])


def test_runtime_canary_allows_only_predeclared_float_tolerance(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canary_run(reference, [_canary_row("development-000")], done=False)
    changed = _canary_row("development-000")
    changed["executed_action_sha256"] = "x" * 64
    _write_canary_run(candidate, [changed], done=True)
    trace = candidate / "behavior_traces" / "ep000.npz"
    np.savez_compressed(
        trace,
        executed_actions=np.asarray([[1.0 + 1e-7, 2.0]], dtype=np.float32),
        plan_boundaries=np.asarray([0, 1], dtype=np.int32),
    )
    report = compare_runs(reference, candidate, min_speedup=0.0, float_atol=1e-6)
    assert report["passed"]
    assert 0.0 < report["max_abs_float_difference"] <= 1e-6

    strict = compare_runs(reference, candidate, min_speedup=0.0, float_atol=1e-8)
    assert not strict["passed"]
    assert any("executed_actions" in error for error in strict["errors"])
