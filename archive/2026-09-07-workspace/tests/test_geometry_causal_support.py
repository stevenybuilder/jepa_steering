"""Support capture checks use anisotropic metrics and tied candidate costs."""
from pathlib import Path
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "geometry_map"))
from capture_causal_support import activation_statistics, checked_artifact, compare_float32_replay, ranks_by_step, require_exact


def test_activation_measurements_use_both_saved_coordinate_metrics():
    dtype = torch.float64
    basis = torch.tensor([[1.], [1.], [0.]], dtype=dtype) / 2 ** .5
    scale = torch.tensor([2., .2, 4.], dtype=dtype)
    before = torch.tensor([[1., 2., 3.], [-2., 1., 0.]], dtype=dtype)
    delta = torch.tensor([[1.], [-.5]], dtype=dtype) @ basis.T * scale
    row = activation_statistics(before, before + delta, delta, basis, scale)
    for prefix in ("requested", "realized"):
        assert row[f"{prefix}_raw_orthogonal_l2"].max() < 1e-12
        assert row[f"{prefix}_standardized_orthogonal_l2"].max() < 1e-12
    torch.testing.assert_close(row["realized_delta"], delta)
    torch.testing.assert_close(row["before_pooled"], before, atol=0, rtol=0)
    # This displacement lies entirely outside both raw and standardized spans.
    off = torch.tensor([[0., 0., 4.]], dtype=dtype).expand(2, 3)
    row = activation_statistics(before, before + off, off, basis, scale)
    torch.testing.assert_close(row["realized_raw_orthogonal_l2"], torch.full((2,), 4., dtype=dtype))
    torch.testing.assert_close(row["realized_standardized_orthogonal_l2"], torch.ones(2, dtype=dtype))


def test_identity_support_records_actual_zero_delta():
    before = torch.randn(5, 4)
    row = activation_statistics(before, before, torch.zeros_like(before), torch.eye(4)[:, :2], torch.ones(4))
    assert row["realized_delta"].eq(0).all()
    assert row["requested_raw_orthogonal_l2"].eq(0).all()


def test_ranks_are_per_step_average_tied_ranks():
    costs = torch.tensor([[3., 1., 1., 2.], [0., 4., 2., 3.]])
    torch.testing.assert_close(ranks_by_step(costs), torch.tensor([[4., 1.5, 1.5, 3.], [1., 4., 2., 3.]], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="finite"):
        ranks_by_step(torch.tensor([[float("nan")]]))


def test_exact_identity_rejects_even_small_perturbation():
    with pytest.raises(RuntimeError, match="exactly"):
        require_exact(torch.tensor([1.]), torch.tensor([1.000001]), "identity")


def test_cross_process_bound_is_fixed_and_separate_from_bit_exact_identity():
    base = torch.tensor([1.], dtype=torch.float32)
    rounded = base + 2 * torch.finfo(torch.float32).eps
    result = compare_float32_replay(rounded, base, "cost")
    assert result["max_absolute_error"] == 2.384185791015625e-7
    with pytest.raises(RuntimeError, match="exactly"):
        require_exact(rounded, base, "identity")
    with pytest.raises(RuntimeError, match="frozen"):
        compare_float32_replay(base + 2e-6, base, "cost")


def test_artifact_hash_checked_before_loading_tensor(tmp_path, monkeypatch):
    (tmp_path / "arm.pt").write_bytes(b"not-a-tensor")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Bad hash tensor opened"))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        checked_artifact(tmp_path, {"complete": True, "outputs": [{"path": "arm.pt", "sha256": "bad"}]}, "arm.pt")
