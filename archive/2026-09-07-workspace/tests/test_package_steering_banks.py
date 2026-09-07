"""Replay-prefix boundaries, sealed access, frozen identities and command metrics."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from package_steering_banks import (expected_split, load_development, motion_summary,
                                  pointer, replay_prefixes, sha, verify_entry)


def test_reach_prefix_excludes_one_warmup_step():
    replans = [{"elapsed_steps": 1, "executed_action_count": 15},
               {"elapsed_steps": 16, "executed_action_count": 9}]
    assert replay_prefixes(replans, 24, "reach_wall") == [(0, 15, 1), (15, 9, 16)]


def test_push_prefix_starts_at_zero_and_covers_short_final_chunk():
    replans = [{"elapsed_steps": 0, "executed_action_count": 30},
               {"elapsed_steps": 30, "executed_action_count": 7}]
    assert replay_prefixes(replans, 37, "pusht") == [(0, 30, 0), (30, 7, 30)]


def test_bad_elapsed_or_missing_action_chunk_rejected():
    with pytest.raises(RuntimeError, match="elapsed/prefix"):
        replay_prefixes([{"elapsed_steps": 0, "executed_action_count": 15}], 15, "reach_wall")
    with pytest.raises(RuntimeError, match="cover"):
        replay_prefixes([{"elapsed_steps": 1, "executed_action_count": 15}], 20, "reach_wall")


def test_held_bank_refused_before_hash_or_tensor_access(tmp_path):
    row = {"task": "reach_wall", "episode": 12, "split": "evaluation", "sealed": True, "path": "does-not-exist.pt"}
    with pytest.raises(RuntimeError, match="sealed evaluation"):
        load_development(tmp_path, row, loader=lambda _: pytest.fail("Held tensor opened"))
    row.update({"sealed": False, "split": "development"})
    with pytest.raises(RuntimeError, match="sealed evaluation"):
        load_development(tmp_path, row, loader=lambda _: pytest.fail("Mislabelled held tensor opened"))


def test_hash_mismatch_blocks_loader(tmp_path):
    (tmp_path / "bank.pt").write_bytes(b"bad")
    row = {"task": "pusht", "episode": 0, "split": "development", "sealed": False,
           "path": "bank.pt", "sha256": "wrong"}
    with pytest.raises(RuntimeError, match="checksum"):
        load_development(tmp_path, row, loader=lambda _: pytest.fail("Bad checksum loaded"))


def test_receipt_cannot_escape_or_accept_bad_hash(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "bank.pt").write_bytes(b"bank")
    with pytest.raises(RuntimeError, match="escapes"):
        verify_entry(tmp_path, sub, {"path": "../bank.pt", "sha256": "unused"})
    with pytest.raises(RuntimeError, match="Hash mismatch"):
        verify_entry(tmp_path, tmp_path, {"path": "bank.pt", "sha256": "bad"})


def test_original_seed_mismatch_rejected(tmp_path):
    (tmp_path / "bank.pt").write_bytes(b"bank")
    row = {"task": "pusht", "episode": 0, "split": "development", "sealed": False, "path": "bank.pt",
           "sha256": sha(tmp_path / "bank.pt"), "environment_seed": 2026090600, "planner_seed": 91600}
    payload = {"episode": 0, "arm": "unsteered_frozen_jepa_wm", "environment_seed": 1, "planner_seed": 2}
    with pytest.raises(RuntimeError, match="seed mismatch"):
        load_development(tmp_path, row, loader=lambda _: payload)


def test_command_net_norm_is_not_frobenius_or_clipped_control():
    actions = np.array([[2., 0., 0., 1.], [-2., 0., 0., 1.]])
    result = motion_summary(actions, 3)
    assert result["translation_net_norm"] == 0
    assert result["translation_unit_direction"] is None
    assert result["translation_frobenius_norm"] == pytest.approx(np.sqrt(8))
    assert result["maximum_absolute_raw_command"] == 2
    assert result["applied_environment_controls_saved"] is False


def test_split_boundaries_and_relative_pointer_contract():
    assert expected_split("reach_wall", 11) == "development"
    assert expected_split("reach_wall", 12) == "evaluation"
    assert expected_split("pusht", 9) == "development"
    assert expected_split("pusht", 10) == "evaluation"
    with pytest.raises(RuntimeError):
        expected_split("pusht", 21)
    p = pointer({"path": "artifacts/a.pt", "sha256": "abc"}, "replans", 2, "simulator")
    assert p["keys"] == ["replans", 2, "simulator"] and p["bank_path"] == "artifacts/a.pt"
