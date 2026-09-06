from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_merge_captures import merge_captures, sha256_file  # noqa: E402


def _capture(root: Path, pair_id: str, success: int, value: float) -> None:
    (root / "activations").mkdir(parents=True)
    (root / "behavior_traces").mkdir()
    np.savez_compressed(root / "activations" / "ep000.npz", site__mean=np.full((4, 2), value))
    np.savez_compressed(
        root / "behavior_traces" / "ep000.npz",
        simulator_states=np.full((5, 3), value),
        goal_state=np.full(3, value),
        executed_actions=np.full((4, 1), value),
        plan_boundaries=np.arange(5),
    )
    trace_hash = sha256_file(root / "behavior_traces" / "ep000.npz")
    episode = {
        "ep": 0,
        "pair_id": pair_id,
        "success": success,
        "behavior_trace": "behavior_traces/ep000.npz",
        "behavior_trace_sha256": trace_hash,
    }
    (root / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
    (root / "realizations.jsonl").write_text(json.dumps({"ep": 0, "pair_id": pair_id}) + "\n")
    common_meta = {
        "config_sha256": "c", "checkpoint_sha256": "k", "repo_commit": "r",
        "task_cfg": "mw-reach", "model": "dino-wm", "env": "mw",
    }
    index = {
        "meta": common_meta,
        "pre_outcome_window": [0, 3],
        "planner_capture_scope": "all_plan_calls_tagged_by_physical_replan",
        "physical_pooling": {"mean": "x"},
        "n_episodes": 1,
        "n_success": success,
        "extra_forward_s": 0.1,
        "capture_errors": [],
        "episodes": [{"ep": 0, "file": "ep000.npz", "success": success}],
    }
    (root / "activations" / "index.json").write_text(json.dumps(index))
    (root / "DONE.json").write_text("{}")


def test_merge_renumbers_activation_and_trace_rows(tmp_path: Path) -> None:
    first, second, out = tmp_path / "first", tmp_path / "second", tmp_path / "merged"
    _capture(first, "p0", 1, 1.0)
    _capture(second, "p1", 0, 2.0)
    result = merge_captures([first, second], out, tmp_path / "vendor")
    assert result["n_episodes"] == 2 and result["n_success"] == 1 and result["n_failure"] == 1
    index = json.loads((out / "activations" / "index.json").read_text())
    assert [row["ep"] for row in index["episodes"]] == [0, 1]
    assert [row["pair_id"] for row in index["episodes"]] == ["p0", "p1"]
    assert index["meta"]["task"] == "mw-reach"
    episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert episodes[1]["behavior_trace"] == "behavior_traces/ep001.npz"
    with np.load(out / episodes[1]["behavior_trace"]) as trace:
        assert np.all(trace["simulator_states"] == 2.0)
