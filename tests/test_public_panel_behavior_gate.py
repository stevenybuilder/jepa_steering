from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_behavior_gate import HASH_FIELDS, evaluate  # noqa: E402
from public_panel_manifest import SeedAllocator, generate_rows, rows_sha256, write_jsonl  # noqa: E402


def _run(tmp_path: Path, outcomes: list[int]) -> tuple[Path, Path]:
    common = {
        "task": "mw-reach",
        "namespace": "gate-test",
        "checkpoint_sha256": "c" * 64,
        "config_sha256": "f" * 64,
        "repo_commit": "1" * 40,
    }
    manifest = generate_rows(
        split="development",
        count=len(outcomes),
        allocator=SeedAllocator("gate-test"),
        **common,
    )
    manifest_path = tmp_path / "development.jsonl"
    write_jsonl(manifest_path, manifest)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_sha = rows_sha256(manifest)
    hashes = {field: field for field in HASH_FIELDS}
    episodes = [
        {
            **row,
            "success": outcome,
            "n_plan_calls": 7,
            "episode_manifest_sha256": manifest_sha,
            "realization_hashes": hashes,
        }
        for row, outcome in zip(manifest, outcomes)
    ]
    write_jsonl(run_dir / "episodes.jsonl", episodes)
    (run_dir / "DONE.json").write_text(json.dumps({"complete": True, "n_logged": len(outcomes)}))
    return run_dir, manifest_path


def test_behavior_gate_passes_balanced_complete_run(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, [0, 1] * 15)
    result = evaluate(run_dir, manifest)
    assert result["passed"]
    assert result["decision"] == "advance_to_fit"


def test_behavior_gate_fails_class_floor(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, [1] + [0] * 29)
    result = evaluate(run_dir, manifest)
    assert not result["passed"]
    assert result["decision"] == "stop_substrate"
    assert any("success class floor" in error for error in result["errors"])


def test_behavior_gate_fails_manifest_drift(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, [0, 1] * 15)
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    rows[0]["planner_seed"] += 1
    write_jsonl(run_dir / "episodes.jsonl", rows)
    result = evaluate(run_dir, manifest)
    assert not result["passed"]
    assert any("planner_seed differs" in error for error in result["errors"])
