from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_frankenstein_gate import main  # noqa: E402


def test_gate_writes_hash_bound_artifact_and_explicit_missing_modules(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(3)
    capture = tmp_path / "capture"
    activation = capture / "activations"
    activation.mkdir(parents=True)
    episodes = []
    for episode in range(10):
        state = rng.normal(size=(5, 4))
        action = np.c_[state[:, 0], state[:, 1]]
        path = activation / f"ep{episode:03d}.npz"
        np.savez_compressed(
            path,
            **{
                "L00.resid_post__mean": state,
                "chosen_actions": action[:, None, :],
            },
        )
        episodes.append({"ep": episode, "file": path.name, "success": episode % 2})
    (capture / "episodes.jsonl").write_text("")
    (activation / "index.json").write_text(json.dumps({
        "episodes": episodes,
        "pre_outcome_window": [0, 5],
    }))
    coordinate = tmp_path / "coordinates.npz"
    np.savez_compressed(
        coordinate,
        site=np.asarray("L00.resid_post"),
        mean=np.zeros(4),
        basis=np.eye(4),
        scale=np.ones(4),
        factor_rotation=np.eye(4),
        representation=np.asarray("global"),
        regime_weights=np.asarray([0.5, 0.5]),
        regime_means=np.asarray([[-1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        regime_var=np.ones((2, 4)),
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "public_panel_frankenstein_gate.py",
        "--capture", str(capture),
        "--coordinates", str(coordinate),
        "--out", str(out),
        "--folds", "2",
        "--sparse-components", "6",
        "--sparse-iterations", "10",
    ])
    assert main() == 0
    report = json.loads((out / "frankenstein_gate.json").read_text())
    assert (out / "frankenstein_modules.npz").is_file()
    with np.load(out / "frankenstein_modules.npz") as artifact:
        assert "attention_hopfield_eligible" in artifact.files
        assert str(artifact["attention_restore_site"].item()) == "L00.attn_out"
        assert float(artifact["sparse_active_epsilon"].item()) == pytest.approx(1e-6)
        assert int(artifact["sparse_ista_iterations"].item()) == 100
    assert report["outcome_labels_used"] is False
    assert len(report["sparse_superposition"]["feature_activation_probability_by_regime"]) == 2
    assert len(report["sparse_superposition"]["feature_regime_mutual_information_nats"]) == 6
    assert report["pattern_separation"]["nuisance_source"] == "episode_progress_only_fallback"
    assert report["pattern_separation"]["privileged_simulator_state_used_online"] is False
    assert report["attention_hopfield"]["decidable"] is False
    assert report["relational_transport"]["decidable"] is False


def test_gate_uses_hash_bound_native_geometry_for_pattern_lures(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(13)
    capture = tmp_path / "capture"
    activation = capture / "activations"
    traces = capture / "behavior_traces"
    activation.mkdir(parents=True)
    traces.mkdir()
    episodes, log_rows = [], []
    for episode in range(10):
        state = rng.normal(size=(5, 4))
        action = np.c_[state[:, 0], state[:, 1]]
        activation_path = activation / f"ep{episode:03d}.npz"
        np.savez_compressed(
            activation_path,
            **{"L00.resid_post__mean": state, "chosen_actions": action[:, None, :]},
        )
        simulator = np.zeros((6, 39), dtype=np.float32)
        simulator[:, :3] = rng.normal(size=(6, 3))
        goal = np.zeros(39, dtype=np.float32)
        goal[-3:] = rng.normal(size=3)
        simulator[:, -3:] = goal[-3:]
        trace_path = traces / f"ep{episode:03d}.npz"
        np.savez_compressed(
            trace_path,
            simulator_states=simulator,
            goal_state=goal,
            plan_boundaries=np.arange(6, dtype=np.int32),
        )
        episodes.append({"ep": episode, "file": activation_path.name, "success": episode % 2})
        log_rows.append({"ep": episode, "behavior_trace": str(trace_path.relative_to(capture))})
    (capture / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in log_rows)
    )
    index_path = activation / "index.json"
    index_path.write_text(json.dumps({"episodes": episodes, "pre_outcome_window": [0, 5]}))
    coordinate = tmp_path / "coordinates.npz"
    np.savez_compressed(
        coordinate,
        site=np.asarray("L00.resid_post"),
        mean=np.zeros(4), basis=np.eye(4), scale=np.ones(4), factor_rotation=np.eye(4),
        representation=np.asarray("global"), regime_weights=np.asarray([0.5, 0.5]),
        regime_means=np.asarray([[-1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        regime_var=np.ones((2, 4)),
    )
    native_json = tmp_path / "model_native_coordinates.json"
    native_json.write_text(json.dumps({
        "capture_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "coordinates": str(coordinate.resolve()),
        "site": "L00.resid_post",
        "adapter": {"task": "mw-reach"},
        "model_native_relational_candidate": True,
    }))
    np.savez_compressed(
        native_json.with_suffix(".npz"),
        site=np.asarray("L00.resid_post"), factor_mean=np.zeros(4), target_mean=np.zeros(3),
        readout_weights=np.zeros((4, 3)), rowspace_basis=np.eye(4)[:, :3],
        eligible=np.asarray(True),
        report_sha256=np.asarray(hashlib.sha256(native_json.read_bytes()).hexdigest()),
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "public_panel_frankenstein_gate.py", "--capture", str(capture),
        "--coordinates", str(coordinate), "--out", str(out), "--folds", "2",
        "--sparse-components", "6", "--sparse-iterations", "10",
        "--coordinate-report", str(native_json),
    ])
    assert main() == 0
    report = json.loads((out / "frankenstein_gate.json").read_text())
    assert report["pattern_separation"]["nuisance_source"] == (
        "audited_simulator_hand_goal_geometry_offline_only"
    )
    assert report["model_native_coordinates"]["eligible"] is True
